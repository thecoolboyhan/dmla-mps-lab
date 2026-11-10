"""
对应网页「工程实训：SFT 模型对话实验」的训练代码块。

训练循环与网页逐行对应，只改下面这些 [本项目适配] 项：
  · 去掉沙箱专属 ProgressReporter / DATA_DIR 注入
  · 设备自动检测 cuda / mps / cpu；BF16 autocast 按后端能力决定
  · DataLoader: pin_memory=False, num_workers=0（跨平台最稳）
  · 数据用 stage5 筛过的短对话子集；seq 768 → 512
  · checkpoint 存 FP32（跨平台互读；MPS/CPU 部分算子不支持 FP16 权重）
  · 增加 loss 曲线落盘、有限值断言、200 步耗时校准

保留（一个没动）：ChatML / labels 掩码、加载 pretrain 权重作为起点、
AdamW、lr=5e-5、线性 warmup 10% + 余弦衰减、梯度裁剪 1.0、
梯度累积、周期保存、epoch 保存、最终保存、seed=42。
"""
import math
import time

import torch
import torch.optim as optim
from contextlib import nullcontext
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import PY_CMD
from config import (CHATML_TEMPLATE, DEVICE, DEVICE_TYPE, FINAL_CKPT,
                    SEED, SFT_ACCUMULATION_STEPS, SFT_BASE,
                    SFT_BATCH_SIZE, SFT_EFFECTIVE_BATCH, SFT_FINAL_CKPT,
                    SFT_GRAD_CLIP, SFT_HIDDEN, SFT_LAYERS, SFT_LEARNING_RATE,
                    SFT_LIMIT_STEPS, SFT_LOG_INTERVAL, SFT_LOSS_LOG,
                    SFT_MAX_SEQ_LEN, SFT_NUM_EPOCHS, SFT_OFFICIAL_PRETRAIN,
                    SFT_SAVE_DIR, SFT_SAVE_INTERVAL, SFT_SUBSET, SFT_WARMUP_RATIO,
                    TOKENIZER_DIR, USE_BF16, sft_banner)
from shared.mini_mind_config import MiniMindConfig, MiniMindForCausalLM
from shared.sft_dataset import SFTDataset

SAVE_DIR = SFT_SAVE_DIR
sft_banner()

# ========== 1. 初始化环境 ==========
device = DEVICE
print(f"[1/5] 设备 {device}  |  BF16 autocast {USE_BF16}")
torch.manual_seed(SEED)

# ========== 2. 加载分词器和 SFT 数据 ==========
print("[2/5] 加载分词器和 SFT 对话数据 ...")
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
tokenizer.chat_template = CHATML_TEMPLATE
if not SFT_SUBSET.exists():
    raise SystemExit("找不到 sft_subset.jsonl，请先运行 stage5_sft_data.py")
train_ds = SFTDataset(str(SFT_SUBSET), tokenizer, max_length=SFT_MAX_SEQ_LEN)
print(f"训练样本数: {len(train_ds):,}")

train_loader = DataLoader(
    train_ds, batch_size=SFT_BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=False, drop_last=True,     # [本项目适配]
)
steps_per_epoch = len(train_loader)
full_total_steps = SFT_NUM_EPOCHS * steps_per_epoch
# LAB_SFT_LIMIT_STEPS 只截断实际步数，不影响 warmup/余弦的分母
total_steps = min(full_total_steps, SFT_LIMIT_STEPS) if SFT_LIMIT_STEPS else full_total_steps
print(f"每 epoch 步数 {steps_per_epoch:,} | 总步数 {total_steps:,} | "
      f"参数更新约 {total_steps//SFT_ACCUMULATION_STEPS:,} 次 | 等效批 {SFT_EFFECTIVE_BATCH}")

# ========== 3. 创建模型并加载预训练权重（网页：SFT 的起点） ==========
print("[3/5] 创建模型并加载预训练权重 ...")
base_path = SFT_OFFICIAL_PRETRAIN if SFT_BASE == "official" else FINAL_CKPT
if not base_path.exists():
    raise SystemExit(f"起点权重不存在: {base_path}\n"
                     f"请先运行 stage5_sft_data.py（official）或 stage3_train.py（local）")
lm_config = MiniMindConfig(hidden_size=SFT_HIDDEN, num_hidden_layers=SFT_LAYERS)
model = MiniMindForCausalLM(lm_config)
weights = torch.load(base_path, map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(weights, strict=False)
print(f"  起点权重 {base_path.name}  ({base_path.stat().st_size/1024/1024:.1f} MB, "
      f"dtype={next(iter(weights.values())).dtype})")
print(f"  结构匹配 missing={len(missing)} unexpected={len(unexpected)}")
if missing or unexpected:
    raise SystemExit("权重结构与模型不匹配：official 档需 768/8 层，local 档需与 stage3 同一个 PRESET")
del weights
print(f"  模型参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
      f"| 权重以 FP32 加载（不使用 .half()，跨平台互读）")
model = model.to(device)

# ========== 4. 配置训练组件 ==========
print("[4/5] 配置优化器和学习率调度 ...")
autocast_ctx = (torch.amp.autocast(DEVICE_TYPE, dtype=torch.bfloat16)
                if USE_BF16 else nullcontext())
optimizer = optim.AdamW(model.parameters(), lr=SFT_LEARNING_RATE)


def get_lr(current_step, total, lr):
    """线性 warmup + 余弦衰减：前 10% 线性升温，之后衰减到初始 lr 的 10%（网页原函数）"""
    warmup_steps = int(SFT_WARMUP_RATIO * total)
    if current_step < warmup_steps:
        return lr * current_step / max(warmup_steps, 1)
    p = (current_step - warmup_steps) / max(total - warmup_steps, 1)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))


# 网页 SFT 的 lr 比预训练低一个数量级，避免破坏已学到的语言能力（灾难性遗忘）
# 注意：网页用 global_step（每个 mini-batch）算调度，梯度累积下粒度更细，行为保持原样。

# ========== 5. 开始训练 ==========
print("[5/5] 开始 SFT 训练 ...")
print("  观察重点：loss 只在 assistant 回答上计算，起点已是会说话的模型，")
print("            因此初始 loss 远低于预训练的 ln(6400)≈8.76\n")
SFT_LOSS_LOG.write_text("step\tepoch\tloss\tlr\telapsed_s\n", encoding="utf-8")

global_step = 0
running_loss, log_count = 0.0, 0
train_start = time.time()
calibrated = False


def save_ckpt(name):
    # [本项目适配] 网页存 FP16；改存 FP32 以便跨平台互读（MPS/CPU 部分算子不支持 FP16 权重）
    path = SAVE_DIR / f"sft_{SFT_BASE}_{name}.pth"
    state_dict = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, path)
    del state_dict
    return path


stop = False
for epoch in range(SFT_NUM_EPOCHS):
    if stop:
        break
    model.train()
    epoch_start = time.time()
    for step, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        lr = get_lr(global_step, full_total_steps, SFT_LEARNING_RATE)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss / SFT_ACCUMULATION_STEPS
        loss.backward()

        if (step + 1) % SFT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), SFT_GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        current_loss = loss.item() * SFT_ACCUMULATION_STEPS
        if not math.isfinite(current_loss):
            raise SystemExit(f"step {global_step} 出现非有限 loss={current_loss}，请降低学习率")
        running_loss += current_loss
        log_count += 1
        global_step += 1

        if global_step % SFT_LOG_INTERVAL == 0:
            avg = running_loss / log_count
            elapsed = time.time() - train_start
            eta = elapsed / global_step * (total_steps - global_step) / 60
            print(f"Epoch[{epoch+1}/{SFT_NUM_EPOCHS}] Step[{global_step}/{total_steps}]  "
                  f"loss={avg:.4f}  lr={lr:.2e}  eta={eta:.1f}min", flush=True)
            with SFT_LOSS_LOG.open("a", encoding="utf-8") as f:
                f.write(f"{global_step}\t{epoch+1}\t{avg:.6f}\t{lr:.8f}\t{elapsed:.1f}\n")
            running_loss, log_count = 0.0, 0

        if not calibrated and global_step >= 200:
            elapsed = time.time() - train_start
            est = elapsed / global_step * full_total_steps / 60
            print(f"  [校准] 前 200 步耗时 {elapsed:.0f}s "
                  f"→ 完整 {full_total_steps:,} 步预计约 {est:.0f} 分钟"
                  f"（无风扇机型按 ×1.2~1.4 估）", flush=True)
            calibrated = True

        if global_step % SFT_SAVE_INTERVAL == 0:
            p = save_ckpt(f"step{global_step}")
            print(f"  -> 保存 checkpoint: {p.name}", flush=True)

        del input_ids, labels, res, loss

        if SFT_LIMIT_STEPS and global_step >= SFT_LIMIT_STEPS:
            stop = True
            break

    p = save_ckpt(f"epoch{epoch+1}")
    print(f"Epoch {epoch+1} 完成，耗时 {(time.time()-epoch_start)/60:.1f} min，已保存 {p.name}\n",
          flush=True)

if running_loss > 0:
    tail_avg = running_loss / log_count
    with SFT_LOSS_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{global_step}\t{epoch+1}\t{tail_avg:.6f}\t"
                f"{get_lr(global_step, full_total_steps, SFT_LEARNING_RATE):.8f}\t"
                f"{time.time()-train_start:.1f}\n")
    print(f"末尾 {log_count} 步平均 loss: {tail_avg:.4f}")

state_dict = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
torch.save(state_dict, SFT_FINAL_CKPT)
print(f"\nSFT 完成：{global_step} 步 / {(time.time()-train_start)/60:.1f} 分钟")
print(f"最终权重: {SFT_FINAL_CKPT}")
print(f"loss 曲线: {SFT_LOSS_LOG}")
print(f"\n下一步： {PY_CMD} stage7_sft_chat.py")
