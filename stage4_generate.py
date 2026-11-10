"""
对应网页「第四阶段：推理与对话 / 文本续写」。

保留网页的 4 个 prompt 和 generate 的全部采样参数，只做三处 [本项目适配]：
  · 设备自动检测 cuda / mps / cpu
  · 不调用 model.half()（FP32 权重跨平台互读，MPS/CPU 部分算子不支持 FP16）
  · 权重路径指向 stage3 保存的 checkpoint
另外加两个教学观察：训练前后的生成对比、loss 曲线的纯文本可视化。
"""
import torch
from transformers import AutoTokenizer

from config import PY_CMD
from config import (DEVICE, FINAL_CKPT, HIDDEN_SIZE, LOSS_LOG, NUM_LAYERS,
                    PRESET, SAVE_DIR, TOKENIZER_DIR, banner)
from shared.mini_mind_config import MiniMindConfig, MiniMindForCausalLM

banner()

print("[1/4] loss 曲线回顾")
if LOSS_LOG.exists():
    rows = [l.split("\t") for l in LOSS_LOG.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]
    if rows:
        first, last = float(rows[0][2]), float(rows[-1][2])
        print(f"  记录点 {len(rows)} 个  |  首个 {first:.3f}  →  末个 {last:.3f}  |  降幅 {first-last:.3f}")
        vals = [float(r[2]) for r in rows]
        if len(vals) >= 2:
            lo, hi = min(vals), max(vals)
            bars = "▁▂▃▄▅▆▇█"
            spark = "".join(bars[min(7, int((v - lo) / max(hi - lo, 1e-6) * 7))] for v in vals)
            print(f"  曲线（左=早期，右=末期，低=更好）: {spark}")
            print("  → 初期陡降 = 快速学会 token 频率分布；后期趋缓 = 余弦 LR 下降 + 需要更多 token 才能继续改善")
        else:
            print("  （仅 1 个记录点：看起来是短程冒烟测试。正式训练会有几十个记录点）")
else:
    print("  未找到 loss_log.tsv，请先运行 stage3_train.py")

print("\n[2/4] 加载分词器和权重")
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
ckpt = FINAL_CKPT
if not ckpt.exists():
    cands = sorted(SAVE_DIR.glob(f"pretrain_{PRESET}_*.pth"))
    if not cands:
        raise SystemExit("没有找到任何 checkpoint，请先运行 stage3_train.py")
    ckpt = cands[-1]
print(f"  权重文件: {ckpt.name}  ({ckpt.stat().st_size/1024/1024:.1f} MB)")

config = MiniMindConfig(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS)
model = MiniMindForCausalLM(config)
weights = torch.load(ckpt, map_location="cpu", weights_only=True)
try:
    model.load_state_dict(weights, strict=False)
except RuntimeError as e:
    raise SystemExit(f"权重与当前 PRESET 的模型结构不匹配，请确认训练时用的是同一个 PRESET。\n{e}")
model = model.to(DEVICE).eval()                  # [本项目适配] 网页这里是 .half().to(device)
print(f"  模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  |  设备 {DEVICE}")

test_prompts = [
    "人工智能是",
    "深度学习在自然语言处理中的应用包括",
    "The transformer architecture is",
    "机器学习模型训练的关键步骤是",
]


def generate(model, prompt, label):
    inputs = tokenizer(tokenizer.bos_token + prompt, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        try:
            out = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64,
                temperature=0.85, top_p=0.85, top_k=50, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
        except (RuntimeError, TypeError):
            out = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64, do_sample=True, temperature=0.85, top_p=0.85,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"    {label}: {text[:120]}")


print("\n[3/4] 预训练模型文本续写（网页第四阶段原样）")
print("=" * 68)
for prompt in test_prompts:
    print(f"输入: {prompt}")
    generate(model, prompt, "续写")
    print("-" * 68)

print("\n[4/4] 对照：未经训练的同一模型会输出什么")
print("=" * 68)
print("  ⚠ 下面这段【必然是乱码】，是脚本故意打印的对照组，不是你训练出来的结果。")
print("    判断训练是否成功，请看上一段 [3/4] 的「续写」，以及 loss_log.tsv 的曲线。")
print("=" * 68)
fresh = MiniMindForCausalLM(MiniMindConfig(hidden_size=HIDDEN_SIZE,
                                           num_hidden_layers=NUM_LAYERS)).to(DEVICE).eval()
for prompt in test_prompts[:2]:
    print(f"输入: {prompt}")
    generate(fresh, prompt, "随机初始化")
    print("-" * 68)

print("\n学习检查：")
print("  · 训练后的续写应出现与 prompt 相关的常见中文词；随机初始化应基本是乱码")
print("  · 若续写反复重复同一短语：这是降配后欠训练 + 采样的正常现象，")
print("    不是 bug。可以回到 config.py 增大 subset_samples 或 epoch 再跑一次对比")
print("  · 试试改 learning_rate 或数据规模，比较两条 loss 曲线 —— 这才是这一章的实验重点")
print("\n⚠ 预训练模型只会「续写」，不会「对话」——网页第四阶段源码注释也是这么写的。")
print("  想让模型会说话，继续跑网页下一章（SFT 模型对话实验）的降配版：")
print(f"    {PY_CMD} stage5_sft_data.py")
print(f"    {PY_CMD} stage6_sft_train.py")
print(f"    {PY_CMD} stage7_sft_chat.py")
print("  详见 GUIDE.md 第九节。")
