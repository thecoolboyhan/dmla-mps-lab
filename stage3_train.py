"""
对应网页「第三阶段：预训练」。训练循环与网页代码块逐行对应，
只改动下面这些「运行环境」相关的部分（每项都标了 [本项目适配]）：
  · 去掉沙箱专属的 ProgressReporter / DATA_DIR 注入
  · 设备自动检测 cuda / mps / cpu；BF16 autocast 按后端能力决定
  · DataLoader: pin_memory=False, num_workers=0（跨平台最稳）
  · checkpoint 存 FP32（跨平台互读；MPS/CPU 部分算子不支持 FP16 权重）
  · 增加 loss 曲线落盘、有限值断言、200 步耗时校准

保留（一个没动）：BF16 混合精度、梯度累积、余弦学习率、
梯度裁剪 1.0、周期保存、epoch 保存、最终保存、seed=42。
"""
import math
import time

import torch
import torch.optim as optim
from contextlib import nullcontext
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import PY_CMD
from config import (ACCUMULATION_STEPS, BATCH_SIZE, DEVICE, DEVICE_TYPE, FINAL_CKPT,
                    GRAD_CLIP, HIDDEN_SIZE, LEARNING_RATE, LIMIT_STEPS, LOG_INTERVAL,
                    LOSS_LOG, MAX_SEQ_LEN, NUM_EPOCHS, NUM_LAYERS, NUM_WORKERS,
                    PRESET, SAVE_DIR, SAVE_INTERVAL, SEED, TOKENIZER_DIR, TRAIN_SUBSET,
                    USE_BF16, banner, compare_with_web)
from shared.mini_mind_config import MiniMindConfig, MiniMindForCausalLM
from shared.pretrain_dataset import PretrainDataset

banner()
compare_with_web()

# ========== 1. 初始化环境 ==========
device = DEVICE                                     # 网页是 cuda if available else cpu；本项目自动检测三种后端
print(f"[1/5] 设备: {device}  |  BF16 autocast: {USE_BF16}")
torch.manual_seed(SEED)

# ========== 2. 加载分词器和数据 ==========
print("[2/5] 加载分词器和训练数据 ...")
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
train_ds = PretrainDataset(str(TRAIN_SUBSET), tokenizer, max_length=MAX_SEQ_LEN)
print(f"训练样本数: {len(train_ds):,}")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False, drop_last=True   # [本项目适配]
)
total_steps_per_epoch = len(train_loader)
full_total_steps = NUM_EPOCHS * total_steps_per_epoch      # 完整训练应有的步数
# LAB_LIMIT_STEPS 只截断「实际跑多少步」，不影响余弦调度的分母，
# 这样短程冒烟/校准时的学习率曲线与正式训练一致
total_steps = min(full_total_steps, LIMIT_STEPS) if LIMIT_STEPS else full_total_steps
print(f"每 epoch 步数: {total_steps_per_epoch:,}")
print(f"总训练步数: {total_steps:,}  |  参数更新次数约 {total_steps//ACCUMULATION_STEPS:,}"
      f"  |  等效批 {BATCH_SIZE*ACCUMULATION_STEPS}")

# ========== 3. 创建模型 ==========
print("[3/5] 创建 Transformer 模型 ...")
lm_config = MiniMindConfig(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS)
model = MiniMindForCausalLM(lm_config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# ========== 4. 配置训练组件 ==========
print("[4/5] 配置优化器和学习率调度 ...")
# [本项目适配] 网页是 cuda → autocast(bf16)，cpu → nullcontext；MPS 同样支持 bf16 autocast
autocast_ctx = (torch.amp.autocast(DEVICE_TYPE, dtype=torch.bfloat16)
                if USE_BF16 else nullcontext())
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def get_lr(current_step, total, lr):
    """余弦学习率调度：平滑衰减到初始值的 10%（网页原函数，未改动）"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total)))


# 注：网页用 global_step（每个 mini-batch 递增）算余弦，而不是优化器更新次数。
# 梯度累积下两者不同，调度粒度更细——这是一个值得留意的工程细节，行为保持原样。

# ========== 5. 开始训练 ==========
print("[5/5] 开始训练 ...")
print("  观察重点：loss 应从 ln(6400)≈8.76 明显下降；初期陡降、后期趋缓对应余弦调度\n")
LOSS_LOG.write_text("step\tepoch\tloss\tlr\telapsed_s\n", encoding="utf-8")

global_step = 0
running_loss, log_count = 0.0, 0
train_start = time.time()
calibrated = False


def save_ckpt(name):
    # [本项目适配] 网页用 v.half().cpu()；改存 FP32 以便跨平台互读（MPS/CPU 部分算子不支持 FP16 权重）
    path = SAVE_DIR / f"pretrain_{PRESET}_{name}.pth"
    state_dict = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, path)
    del state_dict
    return path


stop = False
for epoch in range(NUM_EPOCHS):
    if stop:
        break
    model.train()
    epoch_start = time.time()
    for step, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        lr = get_lr(global_step, full_total_steps, LEARNING_RATE)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss / ACCUMULATION_STEPS
        loss.backward()

        if (step + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        current_loss = loss.item() * ACCUMULATION_STEPS
        if not math.isfinite(current_loss):
            raise SystemExit(f"step {global_step} 出现非有限 loss={current_loss}，请改用 FP32 或降低学习率")
        running_loss += current_loss
        log_count += 1
        global_step += 1

        if global_step % LOG_INTERVAL == 0:
            avg = running_loss / log_count
            elapsed = time.time() - train_start
            eta = elapsed / global_step * (total_steps - global_step) / 60
            print(f"Epoch[{epoch+1}/{NUM_EPOCHS}] Step[{global_step}/{total_steps}]  "
                  f"loss={avg:.4f}  lr={lr:.2e}  eta={eta:.1f}min", flush=True)
            with LOSS_LOG.open("a", encoding="utf-8") as f:
                f.write(f"{global_step}\t{epoch+1}\t{avg:.6f}\t{lr:.8f}\t{elapsed:.1f}\n")
            running_loss, log_count = 0.0, 0

        if not calibrated and global_step >= 200:
            elapsed = time.time() - train_start
            est = elapsed / global_step * full_total_steps / 3600
            print(f"  [校准] 前 200 步耗时 {elapsed:.0f}s "
                  f"→ 完整 {full_total_steps:,} 步预计约 {est:.2f} 小时"
                  f"（本机无风扇，实际按 ×1.2~1.4 估）", flush=True)
            calibrated = True

        if global_step % SAVE_INTERVAL == 0:
            p = save_ckpt(f"step{global_step}")
            print(f"  -> 保存 checkpoint: {p.name}", flush=True)

        del input_ids, labels, res, loss

        if LIMIT_STEPS and global_step >= LIMIT_STEPS:
            stop = True
            break

    if not stop or LIMIT_STEPS:
        p = save_ckpt(f"epoch{epoch+1}")
        print(f"Epoch {epoch+1} 完成，耗时 {(time.time()-epoch_start)/60:.1f} min，已保存 {p.name}\n",
              flush=True)

# 收尾：把最后一个不足 LOG_INTERVAL 的窗口也写进曲线，短程冒烟测试同样有数据
if running_loss > 0:
    tail_avg = running_loss / log_count
    with LOSS_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{global_step}\t{epoch+1}\t{tail_avg:.6f}\t"
                f"{get_lr(global_step, full_total_steps, LEARNING_RATE):.8f}\t"
                f"{time.time()-train_start:.1f}\n")
    print(f"末尾 {log_count} 步平均 loss: {tail_avg:.4f}")

state_dict = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
torch.save(state_dict, FINAL_CKPT)
print(f"训练结束：{global_step} 步 / {(time.time()-train_start)/60:.1f} 分钟")
print(f"最终权重: {FINAL_CKPT}")
print(f"loss 曲线: {LOSS_LOG}")
print(f"\n下一步： {PY_CMD} stage4_generate.py")
