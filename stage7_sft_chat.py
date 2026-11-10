"""
对应网页「工程实训：SFT 模型对话实验」的推理代码块（对话服务）。

保留网页的 ChatML 模板、apply_chat_template、chat() 结构和全部采样参数，
只做 [本项目适配]：设备自动检测（cuda/mps/cpu）、不调用 .half()、权重指向 stage6 产物。

额外加三个教学对照，把网页「SFT 使模型行为发生质变」这句话变成可观察的现象：
  A. 同一个起点权重、SFT 之前：用 BOS+文本续写，看它只会接话不会回答
  B. SFT 之后：用 ChatML 提问，看它是否给出助手式回答
  C. 可选：官方 full_sft_768.pth（网页标准答案），看完整训练的效果
"""
import torch
from transformers import AutoTokenizer

from config import (CHATML_TEMPLATE, DEVICE, SFT_BASE, SFT_BASE_CKPT,
                    SFT_FINAL_CKPT, SFT_HIDDEN, SFT_LAYERS, SFT_LOSS_LOG,
                    SFT_OFFICIAL_SFT, SFT_OFFICIAL_SFT_URL, SFT_SAVE_DIR,
                    TOKENIZER_DIR, sft_banner)
from shared.mini_mind_config import MiniMindConfig, MiniMindForCausalLM

sft_banner()

print("[1/4] SFT loss 曲线回顾")
if SFT_LOSS_LOG.exists():
    rows = [l.split("\t") for l in SFT_LOSS_LOG.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
    if rows:
        first, last = float(rows[0][2]), float(rows[-1][2])
        print(f"  记录点 {len(rows)} 个 | 首个 {first:.3f} → 末个 {last:.3f} | 降幅 {first-last:.3f}")
        print("  → SFT 的 loss 起点远低于预训练（≈8.76），因为起点已是会说话的模型，")
        print("    且只有 assistant 回答部分参与 loss。")
else:
    print("  未找到 sft_loss_log.tsv，请先运行 stage6_sft_train.py")

print("\n[2/4] 加载分词器与 SFT 权重")
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
tokenizer.chat_template = CHATML_TEMPLATE

ckpt = SFT_FINAL_CKPT
if not ckpt.exists():
    cands = sorted(SFT_SAVE_DIR.glob(f"sft_{SFT_BASE}_*.pth"))
    if not cands:
        raise SystemExit("没有找到 SFT checkpoint，请先运行 stage6_sft_train.py")
    ckpt = cands[-1]
print(f"  权重文件: {ckpt.name}  ({ckpt.stat().st_size/1024/1024:.1f} MB)")


def build_model():
    cfg = MiniMindConfig(hidden_size=SFT_HIDDEN, num_hidden_layers=SFT_LAYERS)
    return MiniMindForCausalLM(cfg)


def load(path):
    m = build_model()
    w = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = m.load_state_dict(w, strict=False)
    if missing or unexpected:
        raise SystemExit(f"权重结构不匹配 (missing={len(missing)}, unexpected={len(unexpected)})，"
                         f"请确认 SFT_BASE / PRESET 与训练时一致")
    return m.to(DEVICE).eval()


model = load(ckpt)
print(f"  模型参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f}M | 设备 {DEVICE}")

QUESTIONS = [
    "你好，请介绍一下你自己",
    "太阳为什么从东边升起？",
    "1+1等于几？",
    "机器学习模型训练的关键步骤是什么？",
]


def chat(m, q, max_new_tokens=96):
    messages = [{"role": "system", "content": "你是一个有帮助的AI助手。"},
                {"role": "user", "content": q}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to(DEVICE)
    with torch.no_grad():
        out = m.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.85, top_p=0.85, top_k=50, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def continue_text(m, q, max_new_tokens=96):
    """网页第 1 章的续写方式：BOS + 文本，没有 ChatML，没有 assistant 标记。"""
    inputs = tokenizer(tokenizer.bos_token + q, return_tensors="pt",
                       truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = m.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens, temperature=0.85, top_p=0.85, top_k=50,
            do_sample=True, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, repetition_penalty=1.2,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


print("\n[A] 对照：SFT 之前（同一个起点权重，用第 1 章的续写方式提问）")
print("=" * 68)
if SFT_BASE_CKPT.exists():
    base_model = load(SFT_BASE_CKPT)
    for q in QUESTIONS[:2]:
        print(f"输入: {q}")
        print(f"续写: {continue_text(base_model, q)[:160]}")
        print("-" * 68)
    del base_model
else:
    print(f"  （起点权重 {SFT_BASE_CKPT.name} 不存在，跳过）")

print("\n[B] SFT 之后（ChatML 提问，网页本章的对话方式）")
print("=" * 68)
for q in QUESTIONS:
    print(f"输入: {q}")
    print(f"回答: {chat(model, q)[:200]}")
    print("-" * 68)

print("\n[C] 可选对照：官方 full_sft_768.pth（网页标准答案）")
if SFT_OFFICIAL_SFT.exists():
    official = load(SFT_OFFICIAL_SFT)
    for q in QUESTIONS[:2]:
        print(f"输入: {q}")
        print(f"回答: {chat(official, q)[:200]}")
        print("-" * 68)
else:
    print("  未下载。想要这条对照，用任意下载工具（浏览器 / curl / Python 均可）把下面地址的文件")
    print(f"  保存到 {SFT_OFFICIAL_SFT}，再重跑本脚本：")
    print(f"  {SFT_OFFICIAL_SFT_URL}")
    print("  （项目不依赖任何特定下载工具，三个平台都一样）")

print("\n学习检查：")
print("  · [A] 只会顺着你的话往下续写，[B] 会识别 user/assistant 结构并给出回答")
print("    → 这就是网页说的「从续写到对话的转变」，由 labels 掩码 + ChatML 共同造成")
print("  · 回答仍可能有事实错误（如 1+1）：64M 模型世界知识有限，SFT 只赋予对话行为，")
print("    不显著增加知识量与推理能力 —— 网页实验结论原文也是这么写的")
print("  · 想比较降配的影响：SFT_BASE=local 重跑 stage5→7，看欠训练 base 是否学得会格式")
