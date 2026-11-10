# dmla-mps-lab

> 给《设计机器学习应用系统》(DMLA) 读者的补充实验方案：
> **没有 NVIDIA 独立显卡、显存或算力不够时，也能把「预训练 → SFT 对话」两章完整走一遍。**
> macOS / Linux / Windows 通用，设备自动检测 **CUDA > MPS > CPU**；仅用于个人学习，非商业用途。

配套文章：

- [Transformer 模型预训练实验](https://ai.icyfenix.cn/language-models/architecture-basics/llm-pretrain-experiment.html)（第 1 章）
- [工程实训：SFT 模型对话实验](https://ai.icyfenix.cn/language-models/pretraining/llm-sft-experiment.html)（第 2 章）

📄 **与网页原版的逐项差异（环境 / 超参 / 代码 / 能得到什么）见 [COMPARISON.md](COMPARISON.md)**；逐步操作手册见 **[GUIDE.md](GUIDE.md)**；AI 助手接手本项目时请先读 **[AGENTS.md](AGENTS.md)**（约束、不变量与验证方法）。

---

## 一、为什么会有这个项目

DMLA 网页的这两个实验默认跑在「Docker + NVIDIA GPU」沙箱里：预训练章节 6 个可执行代码块中有 5 个标了 `data-gpu-only="true"`，云端 Serverless 端点也因费用原因未开放 GPU。没有 N 卡的机器既点不动这些 Run 按钮，也没有 CUDA / FlashAttention / NVIDIA Container Toolkit 的对应实现。

本项目**不改模型结构、数据集类和训练循环的算法**，只做三件事：

1. **运行后端通用化**：自动检测 `cuda → mps → cpu`，BF16 autocast 按后端能力启用；DataLoader 单进程；checkpoint 存 FP32 便于跨平台互读。
2. **数值降配**：模型 13.52M、序列 256、12 万条语料，把训练从「几十小时」压到「1~2 小时」。
3. **去掉环境依赖**：用 Python 标准库的 HTTP Range 分段下载语料，代替 `dmla data` + Git LFS + Docker + curl。

**这是教学降配方案，不是复现网页结果**：

| 项 | 网页原版 | 本项目 standard 档 |
|---|---|---|
| 模型 | 63.91M（768 / 8 层） | 13.52M（384 / 6 层） |
| 语料 | 约 127 万条 × 2 epoch | 12 万条 × 2 epoch |
| 序列长度 | 512 | 256 |
| 设备 / 耗时 | RTX 5080 约 2.5 h | Apple M5 实测 86 min |
| 最终 loss | 约 1.85 | 实测 3.03 |

目标不是拿到一个好用的模型，而是**跑通完整链路、看懂 loss 曲线、理解欠训练与梯度累积/混合精度/余弦调度**，并亲手观察「续写模型」经 SFT 变成「对话模型」的过程。

## 二、适合谁 / 不适合谁

| 适合 | 不适合 |
|---|---|
| 只有 Apple Silicon Mac，没有 N 卡 | 想复现网页 loss 1.85 / 64M 全量指标 |
| 想按网页章节一步步理解训练流程 | 想得到一个日常可用的聊天助手 |
| 显存不足、只有 CPU，或想在 Windows / Linux 上跑同一套材料 | 需要 CUDA 专属算子 / FlashAttention / 多卡分布式实战 |
| 教学、自学、做对照实验 | 任何商业用途（许可证限制，见文末） |

## 三、快速开始

```bash
git clone https://github.com/thecoolboyhan/dmla-mps-lab.git
cd dmla-mps-lab

bash setup.sh                       # macOS / Linux；建 .venv，不动系统 Python
# Windows PowerShell： powershell -ExecutionPolicy Bypass -File .\setup.ps1

# 第 1 章：预训练（数据由脚本自动下载）
<venv-python> stage0_setup.py       # 下载语料 + 分词器，构建训练子集
<venv-python> stage1_tokenizer.py
<venv-python> stage2_model.py
<venv-python> stage3_train.py       # standard 档 M5 实测约 1.5 小时
<venv-python> stage4_generate.py    # 看续写
<venv-python> play_pretrain.py      # 可选：交互输入一句话，看它接什么

# 第 2 章：SFT，让模型学会对话
<venv-python> stage5_sft_data.py
<venv-python> stage6_sft_train.py   # 约 30 分钟（外推值，完整训练未实测）
<venv-python> stage7_sft_chat.py
```

`<venv-python>` 按平台替换；**脚本运行时打印的「下一步」会自动给出你平台上的正确路径**。

| 操作 | macOS / Linux | Windows PowerShell |
|---|---|---|
| 解释器 | `.venv/bin/python` | `.venv\Scripts\python.exe` |
| 跑脚本 | `.venv/bin/python stage0_setup.py` | `.venv\Scripts\python.exe stage0_setup.py` |
| 切档位 | `PRESET=quick .venv/bin/python stage3_train.py` | `$env:PRESET="quick"; .venv\Scripts\python.exe stage3_train.py` |
| 指定后端 | `LAB_DEVICE=cpu .venv/bin/python stage3_train.py` | `$env:LAB_DEVICE="cpu"; .venv\Scripts\python.exe stage3_train.py` |

后端通常不用管，`config.py` 会自动选；`LAB_DEVICE` / `LAB_BF16` 只在做对照实验时才需要。想先快速验证链路：`PRESET=quick` 跑 `stage3_train.py`，12~18 分钟走完 0→4。

## 四、平台支持

| 平台 | 自动选中 | 状态 |
|---|---|---|
| Apple Silicon Mac（macOS 14+） | MPS + BF16 | ✅ **完整实测**：standard 档 86 分钟 / loss 3.029 |
| Linux / Windows + NVIDIA GPU | CUDA + BF16（新卡） | ✅ 自动启用；本机无 N 卡，**真机未实测** |
| Linux / Windows 纯 CPU | CPU + FP32 | ✅ 链路一致，已做短程实跑验证；耗时慢一个数量级 |
| WSL2 / Git Bash | 按内部环境 | ✅ 可用；命令按 Linux 写法 |
| Intel Mac（macOS x86_64） | — | ⚠️ `torch 2.15.0` 只有 arm64 macOS wheel，需自行改用旧版 / CPU 版 torch |

诚实说明：CUDA 分支的设备选择与 BF16 保护逻辑已实现并做了逻辑测试，但**没有在真实 N 卡上跑过完整训练**；MPS 与 CPU 路径都做了短程实跑验证。如果你在 N 卡上跑，欢迎反馈实际耗时。

## 五、仓库里不含什么

**模型权重、语料、分词器、官方预训练权重都不在仓库里**，全部由脚本按需联网下载（ModelScope 国内镜像）。clone 后目录里不会出现 `.venv/`、`data/`、`checkpoints/`，第一次运行相应脚本时自动创建。

| 文件 | 大小 | 由谁下载 |
|---|---:|---|
| `tokenizer.json` / `tokenizer_config.json` | 约 450 KB | `stage0_setup.py` |
| 预训练语料采样池 + 12 万条子集 | 约 260 MB | `stage0_setup.py` |
| SFT 对话语料 `sft_t2t_tiny.jsonl` | 86 MB | `stage5_sft_data.py` |
| 官方 `pretrain_768.pth` / `full_sft_768.pth` | 各 131 MB | `stage5_sft_data.py` |
| 自己训练出的 checkpoint | 每个约 61 MB | `stage3_train.py` / `stage6_sft_train.py` |

`examples/loss_log_standard_sample.tsv` 是一份真实训练曲线样例（150 个记录点，loss 8.110 → 3.029），不用自己跑也能看曲线形状。

> 为什么 SFT 阶段用官方 `pretrain_768.pth` 当起点：网页第 2 章的起点固定是第 1 章产物（768 / 8 层），而降配版第 1 章训出的是 384 / 6 层，结构接不上。这里把网页第 1 章「应该产出」的文件补齐，之后流程与网页完全一致。想用自己训的权重做对照，设 `SFT_BASE=local`。

## 六、与网页章节的对应

| 网页 | 本项目 |
|---|---|
| 实验准备 · `dmla data` + 文件检查 | `stage0_setup.py`、`stage1_tokenizer.py [1/3]` |
| 第 1 阶段 · 分词器演示 / `PretrainDataset` 定义 | `stage1_tokenizer.py`、`shared/pretrain_dataset.py` |
| 第 2 阶段 · 模型定义 | `stage2_model.py`、`shared/mini_mind_config.py` |
| 第 3 阶段 · 预训练 | `stage3_train.py` |
| 第 4 阶段 · 推理续写 | `stage4_generate.py`、`play_pretrain.py` |
| 第 2 章 · SFT 语料 / ChatML / labels 掩码 | `stage5_sft_data.py`、`shared/sft_dataset.py` |
| 第 2 章 · SFT 训练 | `stage6_sft_train.py` |
| 第 2 章 · 对话服务 | `stage7_sft_chat.py` |

完整的逐代码块映射与每一步该看懂什么，见 [GUIDE.md](GUIDE.md) 和 [COMPARISON.md](COMPARISON.md)。

## 七、环境要求

- **macOS / Linux / Windows** 均可；Python ≥ 3.11（`setup.sh` / `setup.ps1` 用 `uv` 装 3.14）
- `torch==2.15.0`、`transformers==5.22.1`、`tokenizers>=0.22`（三个平台都有官方 wheel）
- 后端自动检测；有 N 卡会自动用 CUDA，Apple Silicon 用 MPS，都没有则 CPU
- 16GB 内存可跑 standard 档；磁盘预留约 2GB 给数据与 checkpoint
- 长时间训练请接电、避免休眠（无风扇 Mac 会热降频，实测 0.277 → 0.395 s/step）
- 语料走 ModelScope 国内镜像，**境外网络访问可能较慢**；纯 CPU 且不想拉 CUDA 运行时，可改用 PyTorch 官方 CPU index 单独装 torch

用 `PRESET=quick / standard / faithful` 切换档位，三档**等效批大小恒为 256**，与网页一致。改数值只需编辑 `config.py`。

## 八、实测记录（Apple M5 / 16GB，其他平台未实测）

| 项目 | 结果 |
|---|---|
| 预训练 standard 档 | 15,000 步，86.3 分钟，平均 0.345 s/step |
| loss 曲线 | 8.110 → **3.029**，降幅 5.08；学习率 1e-3 → 1e-4 余弦衰减 |
| 续写效果 | 中文通顺、有主题词，但内容会胡说、重复；英文 prompt 输出破碎 |
| 权重健康 | `missing=0 / unexpected=0`，FP32，无 NaN/Inf |
| SFT 阶段 | 全链路 20 步冒烟通过（初始 loss 1.906）；**完整 1,875 步训练未跑完**，最终对话质量未经实测，约 30 分钟为前 200 步外推 |

**诚实说明**：预训练模型只会续写、不会对话，这是网页设计如此；SFT 赋予的是「对话行为」而非知识，64M 级模型仍会有事实错误（官方权重实测把 1+1 答成 3）。本项目的终点是教学现象，不是可用助手。

## 九、目录结构

```
├── README.md                 本说明
├── AGENTS.md                 给接手项目的 AI 助手：约束 / 不变量 / 验证方法
├── COMPARISON.md             与网页原版的逐项差异（详细）
├── GUIDE.md                  中文学习手册（逐步操作 / 验收 / 对照实验）
├── NOTICE.md                 上游署名、来源与修改说明
├── LICENSE                   CC BY-NC-SA 4.0
├── config.py                 唯一配置中心（档位 / 路径 / 超参 / 设备检测）
├── setup.sh / setup.ps1      macOS·Linux / Windows 环境安装
├── requirements.txt
├── stage0_setup.py  ...  stage7_sft_chat.py    两章共 8 个阶段脚本
├── play_pretrain.py          交互续写小工具
├── shared/
│   ├── mini_mind_config.py    MiniMind 模型定义（上游原版）
│   ├── pretrain_dataset.py    预训练数据集（上游原版）
│   ├── sft_dataset.py         SFT 数据集（上游改编，仅替换 IO）
│   └── download.py            标准库 HTTP Range 下载器（本项目原创）
└── examples/
    └── loss_log_standard_sample.tsv
```

## 十、许可证与署名（重要）

上游《设计机器学习应用系统》(DMLA) 采用 **CC BY-NC-SA 4.0**，其 README 明确要求衍生作品以**相同许可证**发布，且**不得用于商业用途**。因此本项目同样以 **CC BY-NC-SA 4.0** 发布：

- 本仓库中的模型定义、数据集类、训练循环结构均改编自 DMLA，已完整署名并列明修改，见 [NOTICE.md](NOTICE.md)
- 语料 / 官方权重来自 MiniMind 开源项目（Apache-2.0），经 ModelScope `icyfenix` 镜像分发，同样在 NOTICE 中署名
- **仅供个人学习与非商业教学使用**；转载、改编请保留署名、链接与相同许可证

如果这些条款与你的使用场景不符，请直接阅读原文网页，不要分发本仓库。
