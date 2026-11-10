"""
配置中心：整套材料里唯一需要你修改的文件。

对照网页第 3 阶段「训练超参数」一节，下面每个字段都能在原文找到对应项。
改动只发生在「数值」和「运行设备」上，训练流程与网页代码逐行一致。

切换档位：  PRESET=quick .venv/bin/python stage3_train.py
"""
import os
import logging
import warnings

# transformers 5.x 对 MiniMindConfig 的 rope_scaling 字段有一条无害的弃用提示；
# torch 在 macOS 上也会打印一条 distributed redirects 提示。都过滤掉，保持日志干净。
warnings.filterwarnings("ignore", message="`rope_scaling` is deprecated")
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

# 必须在 import torch 之前设置：允许 MPS 用满统一内存，避免大 batch 被水位线误杀
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAVE_DIR = ROOT / "checkpoints"
# 某些 macOS 沙箱的 sitecustomize 会拦截 Path.mkdir(exist_ok=True) 并误报 EEXIST，
# 这里改成先判断再创建，保证任何环境下都能通过。
for _d in (DATA_DIR, SAVE_DIR):
    if not _d.exists():
        _d.mkdir()

CORPUS_SAMPLE = DATA_DIR / "corpus_sample.jsonl"   # stage0 分段下载的原始采样池
TRAIN_SUBSET = DATA_DIR / "train_subset.jsonl"     # stage0 随机抽样后的训练集
TOKENIZER_DIR = DATA_DIR                            # tokenizer.json / tokenizer_config.json
LOSS_LOG = ROOT / "loss_log.tsv"                   # stage3 的 loss 曲线数据

# 语料与分词器（ModelScope 国内镜像，支持 HTTP Range 分段下载）
BASE_URL = "https://www.modelscope.cn/datasets/icyfenix/Minimind_Pretrain/resolve/master"
CORPUS_URL = f"{BASE_URL}/pretrain_t2t_mini.jsonl"

# ---------- 三档预设 ----------
# 等效批大小恒为 256（batch_size × accumulation_steps），和网页完全一致。
# 耗时为 Apple M5 / 8 核 GPU / 16GB 上用「网页原版代码路径」实测外推的短时峰值，
# 无风扇机型长时间满载请按 ×1.2~1.4 估算。
PRESETS = {
    "quick": dict(
        hidden_size=256, num_hidden_layers=4,      # 4.98M 参数
        max_seq_len=128, batch_size=32, accumulation_steps=8,
        num_epochs=2, learning_rate=1e-3,
        subset_samples=80_000, download_mb=80,
        desc="约 12~18 分钟：跑通全流程，看到 loss 陡降，生成会重复",
    ),
    "standard": dict(
        hidden_size=384, num_hidden_layers=6,      # 13.52M 参数
        max_seq_len=256, batch_size=16, accumulation_steps=16,
        num_epochs=2, learning_rate=1e-3,
        subset_samples=120_000, download_mb=160,
        desc="约 1.5 小时（实测 86 分钟）：推荐档，实测最终 loss 3.03，续写通顺但内容会胡说",
    ),
    "faithful": dict(
        hidden_size=768, num_hidden_layers=8,      # 63.91M 参数，与网页一致
        max_seq_len=256, batch_size=8, accumulation_steps=32,
        num_epochs=1, learning_rate=5e-4,
        subset_samples=100_000, download_mb=160,
        desc="约 2~4 小时：保留原模型规模，参数量/显存表可与网页逐项核对",
    ),
}

PRESET = os.environ.get("PRESET", "standard")
if PRESET not in PRESETS:
    raise SystemExit(f"未知 PRESET={PRESET}，可选：{', '.join(PRESETS)}")
P = PRESETS[PRESET]

HIDDEN_SIZE = P["hidden_size"]
NUM_LAYERS = P["num_hidden_layers"]
MAX_SEQ_LEN = P["max_seq_len"]
BATCH_SIZE = P["batch_size"]
ACCUMULATION_STEPS = P["accumulation_steps"]
NUM_EPOCHS = P["num_epochs"]
LEARNING_RATE = P["learning_rate"]
# 可用环境变量临时覆盖，方便先跑小规模冒烟测试或自行调数据量
SUBSET_SAMPLES = int(os.environ.get("LAB_SUBSET_SAMPLES", P["subset_samples"]))
DOWNLOAD_MB = int(os.environ.get("LAB_DOWNLOAD_MB", P["download_mb"]))

GRAD_CLIP = 1.0            # 网页原值
LOG_INTERVAL = 100         # 网页原值
SAVE_INTERVAL = 2000       # 网页原值 1000，这里调大以免小实验产生过多 checkpoint
SEED = 42                  # 网页原值
NUM_WORKERS = 0            # macOS 上 spawn DataLoader 子进程容易失败，改单进程
EFFECTIVE_BATCH = BATCH_SIZE * ACCUMULATION_STEPS

# 仅供材料自检使用：设为正整数可让 stage3 只跑 N 步，快速验证全链路
LIMIT_STEPS = int(os.environ.get("LAB_LIMIT_STEPS", "0"))

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DEVICE_TYPE = "mps" if DEVICE.type == "mps" else "cpu"

# 网页用 CUDA；MPS 上 BF16 autocast 实测可用且比 FP32 快约 35%，数值有限
USE_BF16 = DEVICE_TYPE == "mps"

FINAL_CKPT = SAVE_DIR / f"pretrain_{PRESET}_final.pth"


def banner():
    print("=" * 68)
    print(f"  档位 {PRESET}  |  {P['desc']}")
    print(f"  设备 {DEVICE}  |  torch {torch.__version__}  |  BF16 autocast {USE_BF16}")
    print(f"  模型 {HIDDEN_SIZE}/{NUM_LAYERS}层  seq={MAX_SEQ_LEN}  "
          f"batch={BATCH_SIZE}×accum={ACCUMULATION_STEPS}={EFFECTIVE_BATCH}")
    print(f"  数据 {SUBSET_SAMPLES:,} 条 × {NUM_EPOCHS} epoch  lr={LEARNING_RATE}")
    print("=" * 68)


def compare_with_web():
    """打印与网页原版的差异，方便边跑边对照"""
    print("\n与网页原版的差异（改的只有这些）：")
    rows = [
        ("hidden_size", 768, HIDDEN_SIZE),
        ("num_hidden_layers", 8, NUM_LAYERS),
        ("max_seq_len", 512, MAX_SEQ_LEN),
        ("batch_size", 32, BATCH_SIZE),
        ("accumulation_steps", 8, ACCUMULATION_STEPS),
        ("num_epochs", 2, NUM_EPOCHS),
        ("learning_rate", "5e-4", LEARNING_RATE),
        ("训练样本数", "约 127 万（全量）", f"{SUBSET_SAMPLES:,}（随机子集）"),
        ("device", "cuda", DEVICE_TYPE),
        ("DataLoader", "pin_memory=True, num_workers=2", f"pin_memory=False, num_workers={NUM_WORKERS}"),
        ("checkpoint", "保存为 FP16 (.half())", "保存为 FP32（MPS 部分算子不支持 FP16）"),
    ]
    for name, web, ours in rows:
        flag = "  " if web == ours else "→ "
        print(f"  {flag}{name:22s} 网页 {str(web):30s} 本机 {ours}")
    print(f"  {'  ':2s}{'等效批大小':22s} 网页 256                           本机 {EFFECTIVE_BATCH}（不变）")
    print()


# =====================================================================
# SFT 阶段：网页下一章「工程实训：SFT 模型对话实验」的降配版
# =====================================================================
# 网页第 1 章的产物 pretrain_768.pth（768/8 层）是第 2 章 SFT 的唯一起点。
# 本机 stage3 训出的是 384/6 层降配模型，结构接不上网页 SFT 章节，
# 因此默认改用作者官方已训好的 pretrain_768.pth —— 它正是网页第 1 章
# 「应该产出」的那个文件，结构/路径/后续流程与网页 100% 对齐。
# 想用自己 stage3 训的权重做对照实验：SFT_BASE=local
SFT_BASE = os.environ.get("SFT_BASE", "official")     # official | local
if SFT_BASE not in ("official", "local"):
    raise SystemExit(f"未知 SFT_BASE={SFT_BASE}，可选 official / local")

_SFT_MS = "https://www.modelscope.cn/models/icyfenix/minimind/resolve/master"
SFT_OFFICIAL_PRETRAIN_URL = f"{_SFT_MS}/pretrain/pretrain_768.pth"
SFT_OFFICIAL_SFT_URL = f"{_SFT_MS}/sft/full_sft_768.pth"
SFT_OFFICIAL_PRETRAIN = DATA_DIR / "official_pretrain_768.pth"
SFT_OFFICIAL_SFT = DATA_DIR / "official_full_sft_768.pth"
SFT_CORPUS_URL = "https://www.modelscope.cn/datasets/icyfenix/Minimind_SFT/resolve/master/sft_t2t_tiny.jsonl"

# 网页 SFT 章固定 768/8；用 local base 时沿用预训练档位的结构
SFT_HIDDEN = 768 if SFT_BASE == "official" else HIDDEN_SIZE
SFT_LAYERS = 8 if SFT_BASE == "official" else NUM_LAYERS

SFT_CORPUS = DATA_DIR / "sft_t2t_tiny.jsonl"      # 下载的 5 万条对话语料
SFT_SUBSET = DATA_DIR / "sft_subset.jsonl"        # 抽取后的训练子集
SFT_SAVE_DIR = ROOT / "sft_checkpoints"
if not SFT_SAVE_DIR.exists():
    SFT_SAVE_DIR.mkdir()
SFT_LOSS_LOG = ROOT / "sft_loss_log.tsv"
SFT_FINAL_CKPT = SFT_SAVE_DIR / f"full_sft_{SFT_BASE}.pth"
SFT_BASE_CKPT = SFT_OFFICIAL_PRETRAIN if SFT_BASE == "official" else FINAL_CKPT

# 网页原值 vs 本机降配（等效批 32 保持不变）
# 实测依据：sft_t2t_tiny 的 token 长度 p50=468 / p90=639。
# seq=256 只能完整保留 8.7% 样本（且几乎都是极短拒答），seq=512 可留 48.8%；
# M5 上 768/8 模型 seq=512 b8 实测 0.915 s/step → 1875 步约 29 分钟。
SFT_SAMPLES = int(os.environ.get("LAB_SFT_SAMPLES", "15000"))   # 网页 5 万
SFT_MAX_SEQ_LEN = 512                    # 网页 768
SFT_BATCH_SIZE = 8                      # 网页 16
SFT_ACCUMULATION_STEPS = 4              # 网页 2 → 等效批仍 32
SFT_NUM_EPOCHS = 1                      # 网页 2
SFT_LEARNING_RATE = 5e-5                # 网页原值（比预训练低一个数量级）
SFT_WARMUP_RATIO = 0.1                  # 网页原值
SFT_GRAD_CLIP = 1.0                     # 网页原值
SFT_LOG_INTERVAL = 50                   # 网页原值
SFT_SAVE_INTERVAL = 1000                # 网页 500，调大减少小实验产物
SFT_LIMIT_STEPS = int(os.environ.get("LAB_SFT_LIMIT_STEPS", "0"))
SFT_EFFECTIVE_BATCH = SFT_BATCH_SIZE * SFT_ACCUMULATION_STEPS

# MiniMind tokenizer 未内置 chat_template，网页手动设置这个 ChatML 模板
CHATML_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message.role }}\n"
    "{{ message.content }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def sft_banner():
    print("=" * 68)
    print(f"  SFT 阶段  |  起点权重 {SFT_BASE}"
          f"  |  模型 {SFT_HIDDEN}/{SFT_LAYERS}层  seq={SFT_MAX_SEQ_LEN}")
    print(f"  数据 {SFT_SAMPLES:,} 条 × {SFT_NUM_EPOCHS} epoch  "
          f"batch={SFT_BATCH_SIZE}×accum={SFT_ACCUMULATION_STEPS}={SFT_EFFECTIVE_BATCH}"
          f"  lr={SFT_LEARNING_RATE}")
    print(f"  设备 {DEVICE}  |  BF16 autocast {USE_BF16}")
    print("=" * 68)
    if SFT_BASE == "official":
        print("  起点 = 官方 pretrain_768.pth（等价于网页第 1 章产物），可正常对话")
    else:
        print("  起点 = 你自己 stage3 训出的降配权重（欠训练，只能学会对话格式）")
    print()
