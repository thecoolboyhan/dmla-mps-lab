"""
对应网页「工程实训：SFT 模型对话实验」的 SFTDataset 代码块。

[本机适配] 只有一处：网页用 `datasets.load_dataset('json', ...)` 读 JSONL，
本机 .venv 没装 datasets（会连带 pyarrow 等一堆依赖），这里换成纯 Python
逐行 json.loads。其余全部与网页逐行一致：

  · ChatML 模板（tokenizer 未内置，需手动设置）
  · create_chat_prompt：apply_chat_template 生成对话文本
  · generate_labels：只把 assistant 回答区间的 token 保留为标签，其余 -100
  · pre_processing_chat：20% 概率补 system prompt
  · 右填充到 max_length，填充位不参与 loss

这是本章与预训练最本质的差别：预训练对**每个位置**都算 next-token loss，
SFT 只对 assistant 的回答算 loss —— 用户说的话不参与拟合。
"""
import json
import os
import random

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """SFT 数据集：将对话数据 tokenize 为 ChatML 格式"""

    # 使用 ChatML 格式：<|im_start|>role\ncontent<|im_end|>\n
    CHATML_TEMPLATE = (
        "{% for message in messages %}<|im_start|>{{ message.role }}\n"
        "{{ message.content }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )

    def __init__(self, jsonl_path, tokenizer, max_length=768):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = tokenizer
        if not tokenizer.chat_template:
            tokenizer.chat_template = self.CHATML_TEMPLATE
        self.max_length = max_length
        self.samples = self._load_jsonl(jsonl_path)   # [本机适配]
        # <|im_start|>assistant\n 的 token ID 序列，用于定位助手回答的起点
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n',
                                add_special_tokens=False).input_ids
        # <|im_end|>\n 的 token ID 序列，用于定位助手回答的终点
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n',
                                add_special_tokens=False).input_ids

    @staticmethod
    def _load_jsonl(path):
        """[本机适配] 等价替换 datasets.load_dataset，顺手丢弃分段下载的半行。"""
        out = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                conv = d.get('conversations')
                if conv:
                    out.append({'conversations': conv})
        return out

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """将对话列表应用 chat template 转为文本"""
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )

    def generate_labels(self, input_ids):
        """生成标签：assistant 回答部分保留原始 ID，其余设为 -100"""
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 右侧填充至固定长度；填充位标签为 -100，不参与 loss
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def pre_processing_chat(conversations, add_system_ratio=0.2):
    """预处理对话数据：概率性添加系统提示词"""
    if not conversations:
        return conversations
    if any(conv.get('tools') for conv in conversations):
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
    ]
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
