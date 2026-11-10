"""
对应网页「第二阶段：模型定义」。

模型代码本身放在 shared/mini_mind_config.py，与网页代码块逐行一致，
这里只做三件事：
  1. 按网页的演示代码创建模型、打印参数量（原样）
  2. 把参数量拆开手算一遍，让你对照网页的结构表
  3. 按网页「训练显存占用预估」的方法，用降配后的 B/S/D/I 重算一遍峰值占用
"""
import math

from config import (BATCH_SIZE, HIDDEN_SIZE, MAX_SEQ_LEN, NUM_LAYERS, banner,
                    compare_with_web)
from shared.mini_mind_config import MiniMindConfig, MiniMindForCausalLM

banner()
compare_with_web()

print("[1/3] 创建模型（网页第二阶段演示代码，原样）")
config = MiniMindConfig(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS)
model = MiniMindForCausalLM(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"词表大小: {config.vocab_size}")
print(f"隐藏层维度: {config.hidden_size}")
print(f"Transformer 层数: {config.num_hidden_layers}")
print(f"注意力头数: {config.num_attention_heads} (Q) / {config.num_key_value_heads} (KV)")
print(f"FFN 中间维度: {config.intermediate_size}")
print(f"词嵌入与输出头共享: {config.tie_word_embeddings}")

print("\n[2/3] 参数量手算（网页表格的推导过程）")
D, I = config.hidden_size, config.intermediate_size
nq, nkv, dh = config.num_attention_heads, config.num_key_value_heads, config.head_dim
V = config.vocab_size
emb = V * D
attn = D * (nq * dh) + D * (nkv * dh) * 2 + (nq * dh) * D
ffn = 3 * D * I
per_layer = attn + ffn
calc = emb + NUM_LAYERS * per_layer
print(f"  词嵌入(与输出头共享)  {V}×{D}                 = {emb/1e6:8.3f}M")
print(f"  每层 Attention        Q+K+V+O                = {attn/1e6:8.3f}M")
print(f"  每层 FFN(SwiGLU)      3×{D}×{I}          = {ffn/1e6:8.3f}M")
print(f"  每层小计                                      = {per_layer/1e6:8.3f}M")
print(f"  {NUM_LAYERS} 层 × {per_layer/1e6:.3f}M + 词嵌入 {emb/1e6:.3f}M     = {calc/1e6:8.3f}M")
print(f"  实际统计 {total_params/1e6:.3f}M（差值来自 RMSNorm 的 scale 参数）")
print(f"  → 把 hidden_size 从 768 降到 {D}，FFN 会自动从 2432 变成 {I}，"
      f"head_dim 从 96 变成 {dh}：这是配置类的公式在起作用")

print("\n[3/3] 显存占用重算（照网页第 3 阶段的方法，代入本机参数）")
B, S = BATCH_SIZE, MAX_SEQ_LEN
static = 4 * total_params * 4 / 1024**3       # 参数 + AdamW m/v + 梯度，均 FP32
ln_out = 4 * B * S * D                        # LayerNorm 输出 ×2，BF16
qkv = B * S * (nq + 2 * nkv) * dh * 2
attn_out = B * S * D * 2
swiglu = 4 * B * S * I * 2
per_layer_act = (ln_out + qkv + attn_out + swiglu) / 1024**3
activations = NUM_LAYERS * per_layer_act
logits = (B * (S - 1) * V * (2 + 2 + 4 + 4)) / 1024**3   # 网页：BF16 原始 + 切片 + FP32 upcast + softmax
runtime = 0.5
peak = (static + activations + logits + runtime) * 1.15    # 碎片化 ~15%
print(f"  B={B} S={S} D={D} I={I} nq={nq} nkv={nkv} dh={dh}")
print(f"  静态占用(参数+优化器+梯度)  {static:.2f} GB   [网页原版 0.95 GB]")
print(f"  前向激活({NUM_LAYERS} 层, Flash Attn)  {activations:.2f} GB   [网页原版 3.31 GB]")
print(f"  Logits + Loss                 {logits:.2f} GB   [网页原版 1.17 GB]")
print(f"  运行时 + 碎片化               ~{runtime:.2f} GB +15%")
print(f"  估算峰值                      {peak:.2f} GB   [网页原版 7.4 GB]")
print(f"  → 你的 16GB 统一内存完全装得下；standard 档实测 MPS 驱动侧占用约 2.2 GB")
print("\n下一步： .venv/bin/python stage3_train.py")
print("（建议先 PRESET=quick 跑一遍全流程，再切 standard 正式训练）")
