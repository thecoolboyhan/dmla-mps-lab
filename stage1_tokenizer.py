"""
对应网页「实验准备」+「第一阶段：分词器」两节。

网页代码原样保留，另外补了一段 Dataset 输出的观察：
一条样本到底长什么样、labels 为什么右移、-100 起什么作用、
以及 padding 明明不参与 loss 却仍然在消耗算力。
"""
import json

from config import (DATA_DIR, MAX_SEQ_LEN, TOKENIZER_DIR, TRAIN_SUBSET, banner)
from transformers import AutoTokenizer

banner()
print("\n[1/3] 文件检查（对应网页「实验准备」代码块）")
for name in ("tokenizer.json", "tokenizer_config.json", "train_subset.jsonl"):
    f = DATA_DIR / name
    state = f"已存在  {f.stat().st_size/1024/1024:.2f} MB" if f.exists() else "未找到"
    print(f"  {name:26s} {state}")
if not TRAIN_SUBSET.exists():
    raise SystemExit("请先运行 stage0_setup.py")

print("\n[2/3] 分词器演示（对应网页「第一阶段」代码块，未改动）")
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
print(f"词表大小: {len(tokenizer)}")
print(f"BOS 标记: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
print(f"EOS 标记: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"PAD 标记: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

text = "大语言模型是人工智能的重要方向"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens, skip_special_tokens=True)
print(f"\n分词演示:")
print(f"原文: {text}")
print(f"Token IDs: {tokens}")
print(f"Token 数量: {len(tokens)}")
print(f"解码还原: {decoded}")
print(f"压缩率: {len(text)/len(tokens):.2f} 字符/token")

print("\n[3/3] 一条训练样本长什么样（补充观察，网页未展开）")
from shared.pretrain_dataset import PretrainDataset
ds = PretrainDataset(str(TRAIN_SUBSET), tokenizer, max_length=MAX_SEQ_LEN)
first_text = json.loads(TRAIN_SUBSET.open(encoding="utf-8").readline())["text"]
print(f"原始文本: {first_text[:80]}...")
ids, labels = ds[0]
print(f"input_ids 形状: {tuple(ids.shape)}  → 固定填充到 max_seq_len={MAX_SEQ_LEN}")
print(f"前 16 个 input_ids: {ids[:16].tolist()}")
print(f"前 16 个 labels   : {labels[:16].tolist()}")

print("\n  ⚠ 这里最容易看错：Dataset 里 labels = input_ids.clone()，两者前 16 位完全相同，")
print("    Dataset 只做了一件事——把 PAD 位置改成 -100，让 cross_entropy 忽略它们。")
print("    真正的「右移一位」发生在模型 forward 里（见 shared/mini_mind_config.py 末尾）：")
print("        x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()")
print("    即：位置 t 的隐状态输出，去预测位置 t+1 的 token —— 这才是 next-token prediction。")
print("    网页第 2 阶段和第 3 阶段之间，这个衔接值得自己找到并核对一遍。")

sample_n = min(200, len(ds))
pad_total = 0
for i in range(sample_n):
    x, _ = ds[i]
    pad_total += int((x == tokenizer.pad_token_id).sum())
pad_n = int((ids == tokenizer.pad_token_id).sum())
mask_n = int((labels == -100).sum())
print(f"\n  本条样本 PAD={pad_n} 个，labels 中 -100={mask_n} 个（两者相等，说明屏蔽生效）")
print(f"  前 {sample_n} 条样本平均 padding 占比: {pad_total/(sample_n*MAX_SEQ_LEN)*100:.1f}%")
print("  → padding 位置不参与 loss，但前向/反向计算照样进行；")
print("    网页原版 seq=512 时全语料平均约 59% 的算力花在这上面")
print("\n下一步： .venv/bin/python stage2_model.py")
