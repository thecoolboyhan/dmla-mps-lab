# 本机降配版 64M Transformer 预训练实验

配套网页：[Transformer 模型预训练实验](https://ai.icyfenix.cn/language-models/architecture-basics/llm-pretrain-experiment.html)

这套材料把网页的四个阶段搬到 Apple Silicon Mac 上按阶段重跑一遍。**模型定义、数据集、训练循环的代码来自网页原版**，只改运行设备和数值；你照着网页一节一节往下读、每节跑对应脚本即可。

网页默认沙箱是「Docker + NVIDIA GPU」，本章全部代码块被标记为 GPU-only，Mac 上点不动；这套材料走 `--native` 等价路线，用 MPS 后端，不需要 Docker。

---

## 一、准备（只需一次）

**本目录已附带一份预建并验证过的 `.venv`**（torch 2.15.0 + transformers 5.22.1 + tokenizers，MPS 可用），直接跳到第二步即可，不用先安装。

换机器、移走目录或需要重建环境时再执行：

```bash
cd dmla-mps-lab
bash setup.sh                    # 建 .venv 并装依赖，不动系统 Python
```

已在 Apple M5 / macOS 27 / Python 3.14 上实测通过。若你已有环境，只要 `torch.backends.mps.is_available()` 为 True 且 transformers 能 import 即可跳过。

后续所有脚本都用 `.venv/bin/python` 运行。`data/` 里已放好两个分词器文件（共约 450 KB），`stage0` 只需下载语料。

---

## 二、五步学习路线（网页第 1 章：预训练）

> 走完这五步，模型只会**续写**、不会**对话**。想让它「会说话」，继续第九节的
> `stage5 → stage6 → stage7`（对应网页第 2 章「SFT 模型对话实验」）。

| 步骤 | 命令 | 对应网页章节 | 耗时 | 你要看懂什么 |
|---|---|---|---:|---|
| 0 | `.venv/bin/python stage0_setup.py` | 实验准备 | 1–3 分钟（下载 ~160MB） | 语料/分词器从哪来；真实 token 长度分布；**为什么 seq=512 有 59% 算力在算 padding** |
| 1 | `.venv/bin/python stage1_tokenizer.py` | 第一阶段：分词器 | 几秒 | BOS/EOS/PAD、压缩率、`-100` 屏蔽、**右移其实在模型 forward 里**、next-token prediction |
| 2 | `.venv/bin/python stage2_model.py` | 第二阶段：模型定义 | 几秒 | GQA / RoPE / SwiGLU / 权重共享；参数量手算；显存估算表怎么用降配参数重算 |
| 3 | `.venv/bin/python stage3_train.py` | 第三阶段：预训练 | 见下表 | BF16、梯度累积、余弦 LR、梯度裁剪、checkpoint；loss 曲线形状 |
| 4 | `.venv/bin/python stage4_generate.py` | 第四阶段：推理续写 | 1–2 分钟 | 训练前后生成对比；欠训练的表现；采样参数的作用 |

**建议顺序**：先用 `PRESET=quick` 把 0→4 完整跑一遍（约 20 分钟，确认链路通、看到 loss 下降），再切 `standard` 正式训练一次。

```bash
PRESET=quick    .venv/bin/python stage3_train.py
PRESET=standard .venv/bin/python stage3_train.py     # 默认档，可省略 PRESET
```

---

## 二·补、照着网页一步步操作（逐代码块对照）

网页本章共 **1 条终端命令 + 6 个「点击 Run」代码块**，其中 5 个标了 `data-gpu-only="true"`，在 Mac 上点不动。下表是逐个等价替换，学习动作一个不少，只是执行方式从「网页 Run」换成「本地终端」。

| # | 网页位置 | 网页动作 | 本机对应 | 当前状态 |
|---|---|---|---|---|
| 0 | 实验准备 | `dmla data`（终端命令，非 Run） | `.venv/bin/python stage0_setup.py` | ✅ 数据已备好，可跳过；重跑走缓存 |
| 1 | 实验准备 | Run ① 文件/语料检查 | `stage0_setup.py [1/4]` 环境自检 + `stage1_tokenizer.py [1/3]` 文件检查 | ✅ 已就绪 |
| 2 | 第一阶段 | Run ② 分词器演示（唯一非 GPU-only） | `stage1_tokenizer.py [2/3]` | ✅ 已实测 |
| 3 | 第一阶段 | Run ③ `PretrainDataset` 类定义 | 读 `shared/pretrain_dataset.py`，用 `stage1_tokenizer.py [3/3]` 观察一条样本 | ✅ 已就绪（网页也注明无需单独执行） |
| 4 | 第二阶段 | Run ④ 模型类定义 | 读 `shared/mini_mind_config.py`，跑 `stage2_model.py` | ✅ 已实测 |
| 5 | 第三阶段 | Run ⑤ 预训练（`data-timeout="unlimited"`） | `.venv/bin/python stage3_train.py` | ⬜ **唯一需要你动手跑的长任务** |
| 6 | 第四阶段 | Run ⑥ 推理续写 | `.venv/bin/python stage4_generate.py` | ⬜ 依赖第 5 步产物 |
| 7 | 实验结论 / 运行结果 | 读网页的 loss 曲线与生成示例 | 看 `loss_log.tsv` + stage4 输出 + 本手册第五节验收清单 | ⬜ 人工对照 |

### 逐步操作

**第 0 步 · 实验准备**（网页「实验准备」）
```bash
cd dmla-mps-lab
.venv/bin/python stage0_setup.py        # 已备好数据，可跳过；重跑不会重复下载
```
看什么：torch / transformers / MPS 是否可用；语料 token 长度 p50/p90；`有效 token 保留率` 与 `padding 计算浪费`。产物：`data/corpus_sample.jsonl`(160M)、`data/train_subset.jsonl`(103M，12 万条)、两个分词器文件。
验收：能说清为什么网页 `seq=512` 时约 59% 算力花在 padding 上。

**第 1 步 · 分词器**（网页「第一阶段：分词器与数据加载」，Run ② ③）
```bash
.venv/bin/python stage1_tokenizer.py
```
看什么：词表 6400、BOS/EOS/PAD 的 id、压缩率 2.14 字符/token；`input_ids` 被固定填充到 256；**前 16 位 `input_ids` 与 `labels` 完全相同**——Dataset 只把 PAD 改成 `-100`，真正的右移在模型 forward 的 `logits[..., :-1, :]` 对 `labels[..., 1:]`。同时打开 `shared/pretrain_dataset.py` 对照阅读（网页 Run ③ 的原码）。
验收：能讲清 next-token prediction 的错位是在哪一行发生的。

**第 2 步 · 模型定义**（网页「第二阶段：模型定义」，Run ④）
```bash
.venv/bin/python stage2_model.py
```
看什么：13.52M 参数的手算拆解、GQA `8Q/4KV`、FFN 自动从 2432 变 1216、显存峰值从网页 7.4GB 降到 1.50GB。同时打开 `shared/mini_mind_config.py` 对照 RoPE / RMSNorm / SwiGLU / 权重共享 / `generate()`。
验收：手算的 13.517M 与脚本的 13.522M 能对上，差值来自 RMSNorm scale。

**第 3 步 · 预训练**（网页「第三阶段：预训练」，Run ⑤）
```bash
# 可选冒烟（约 1 分钟，会写 checkpoint 和 loss_log，正式训练会覆盖）
LAB_LIMIT_STEPS=20 .venv/bin/python stage3_train.py

# 正式训练（standard 档，实测约 86 分钟；接电、别合盖）
.venv/bin/python stage3_train.py
```
看什么：`等效批 256`、`每 epoch 步数 7,500`、第 200 步打印的 `[校准] 完整 15,000 步预计约 x.xx 小时`、每 100 步一行 `loss=... lr=... eta=...`、每 2000 步一次 checkpoint。产物：`checkpoints/pretrain_standard_step*.pth`、`epoch*.pth`、`final.pth`（每个约 61MB，实测合计约 0.6GB）、`loss_log.tsv`（150 个记录点）。
验收：loss 从 ≈8.76 明显下降，曲线「初期陡降、后期趋缓」，能和余弦 LR 对上。

**第 4 步 · 推理续写**（网页「第四阶段：推理与对话」，Run ⑥）
```bash
.venv/bin/python stage4_generate.py
```
看什么：4 个网页原版 prompt 的续写、训练后模型 vs 随机初始化模型的对照、loss 曲线 sparkline。采样参数 `temperature=0.85 / top_p=0.85 / top_k=50 / repetition_penalty=1.2` 与网页一致。
验收：训练后出现与 prompt 相关的中文词，随机初始化基本乱码；续写重复不是 bug，是欠训练。

**第 5 步 · 结论与对照**（网页「实验结论」「运行结果」）
```bash
cat loss_log.tsv
ls -lh checkpoints/
```
按第五节验收清单逐条打勾，再按第六节改 `config.py` 里**一个变量**重跑做对照实验。改之前先 `cp loss_log.tsv loss_log_run1.tsv`。

### 四个不能 100% 复刻的字面操作

| 网页 | 本机 | 影响 |
|---|---|---|
| 点 6 次「Run」按钮（GPU 沙箱） | 终端跑 5 个脚本 | 无，学习动作等价 |
| `dmla data` 下全量 1.2GB + Git LFS | HTTP Range 采样 160MB | 数据分布更窄，教学环节无影响 |
| 768/8 层、127 万条 ×2 epoch | 384/6 层、12 万条 ×2 epoch | **实测 loss 3.03**，达不到网页 1.85 |
| Docker + CUDA + FlashAttention 路径 | MPS + BF16 autocast | 无 CUDA 算子，训练时间量级不同 |

### 切档位的三个注意点

1. `PRESET=quick` 与 `PRESET=standard` 的 checkpoint 按档位命名，互不覆盖；但 **`loss_log.tsv` 是共用的**，会被后一次训练覆盖，做对照实验前先备份。
2. `stage4_generate.py` 必须与训练用**同一个 `PRESET`**，否则 `hidden_size/layers` 对不上会提示权重结构不匹配。
3. 切档位后建议重跑一次 `stage0_setup.py` 以匹配该档的 `subset_samples`；不重跑也能训练，只是沿用上一次抽出的子集。

---

## 三、三档预设

三档的**等效批大小恒为 256**（`batch_size × accumulation_steps`），与网页一致，训练语义不变。耗时是在本机用网页原版代码路径实测外推的短时峰值；MacBook Air 无风扇，长时间满载按 ×1.2–1.4 估。

| 档位 | 模型 | 参数量 | seq | batch×accum | 数据 | epoch | 峰值耗时 | 预期 |
|---|---|---:|---:|---|---|---:|---:|---|
| `quick` | 256 / 4 层 | 4.98M | 128 | 32×8 | 8 万条 | 2 | 12–18 min | loss 陡降；续写会重复 |
| `standard` | 384 / 6 层 | 13.52M | 256 | 16×16 | 12 万条 | 2 | **实测 86 min** | 实测最终 loss **3.03**（网页 1.85，仍欠训练）；中文续写通顺，但内容会胡说、英文较差 |
| `faithful` | 768 / 8 层 | 63.91M | 256 | 8×32 | 10 万条 | 1 | 2–4 h | 参数量/显存表与网页逐项对齐 |

切换方式：`PRESET=quick ...`；改数值请编辑 `config.py`（唯一需要改的文件）。临时覆盖数据量：`LAB_SUBSET_SAMPLES=30000`、下载量 `LAB_DOWNLOAD_MB=80`、只跑 N 步 `LAB_LIMIT_STEPS=30`。

---

## 四、与网页原版的差异

`config.compare_with_web()` 每次运行都会打印这张表。

| 项 | 网页原版 | 本机 | 原因 |
|---|---|---|---|
| hidden_size / layers | 768 / 8 | 384 / 6（standard） | 有限算力下小模型欠训练程度更低，教学现象更完整 |
| max_seq_len | 512 | 256 | 真实语料均值约 210 token，512 时 59% 是 padding |
| batch / accum | 32 / 8 | 16 / 16 | **等效批仍 256**，效果等价，只是更频繁累积梯度 |
| 训练样本 | 约 127 万 × 2 epoch | 12 万 × 2 epoch | 控制在 2 小时量级 |
| lr | 5e-4 | 1e-3 | 小模型收敛更快（faithful 档仍用 5e-4） |
| device | cuda | **mps** | Apple Silicon |
| autocast | `device_type="cuda"`, BF16 | `device_type="mps"`, BF16 | MPS 实测可用，比 FP32 快约 35% |
| DataLoader | `pin_memory=True, num_workers=2` | `False, 0` | macOS spawn 子进程易失败 |
| checkpoint | `.half().cpu()` | `.float().cpu()` | MPS 部分算子不支持 FP16 权重 |
| 进度/路径 | `ProgressReporter`、`DATA_DIR` 注入 | 普通 print、本地相对路径 | 脱离沙箱 |

**没改的**：模型结构、数据集类、BF16 混合精度、余弦调度、梯度裁剪 1.0、梯度累积逻辑、周期保存、seed=42、以及 stage4 的采样参数。

---

## 五、验收清单（判断学习目的是否达到）

- [ ] `stage0` 打印出 token 长度 p50/p90，并能说清 padding 浪费从哪来
- [ ] `stage1` 能说清「Dataset 里 labels 未右移，右移是模型 forward 里 `logits[..., :-1, :]` 对 `labels[..., 1:]` 做的」，以及 `-100` 的作用
- [ ] `stage2` 手算的参数量与脚本打印的 13.52M 对得上；能说清为什么 FFN 变成 1216
- [ ] `stage3` 打印的等效批是 256；loss 从 ≈8.76 降幅 > 3
- [ ] `stage3` 的 loss 曲线呈「初期陡降、后期趋缓」，能把它和余弦 LR 联系起来
- [ ] checkpoint 能保存，`loss_log.tsv` 有数据
- [ ] `stage4` 训练后模型的续写出现与 prompt 相关的中文词，随机初始化模型基本是乱码
- [ ] 能解释「为什么续写会重复」= 降配后欠训练 + 采样退化，而不是当成 bug

---

## 六、值得做的对照实验（这章真正的重点）

跑完 standard 后，改 `config.py` 里**一个变量**再跑一次，比较 `loss_log.tsv` 和 stage4 的续写：

1. `learning_rate` 改 `5e-4` 或 `2e-3` → 看收敛速度与稳定性
2. `SUBSET_SAMPLES` 减半 → 看数据量对最终 loss 与重复现象的影响
3. `max_seq_len` 改 128 → 看长文本信息损失如何反映在 loss 上
4. `accumulation_steps` 改 8（等效批变 128）→ 看等效批大小对曲线平滑度的影响

`quick` 档跑一组只要十几分钟，适合做这种对照。为了不覆盖结果，先 `cp loss_log.tsv loss_log_run1.tsv`。

---

## 七、常见坑

| 现象 | 处理 |
|---|---|
| `MPS 可用 False` | 用 `setup.sh` 建的环境；确认 macOS 与 torch 版本；别在 Rosetta/x86 终端里跑 |
| 训练中途 OOM | `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` 已在 `config.py` 设好；仍不行就把 `batch_size` 减半、`accumulation_steps` 加倍 |
| 出现非有限 loss | 脚本会主动中断；把 `learning_rate` 减半，或临时设 `USE_BF16=False` 用 FP32 |
| 速度与预估不符 | 合盖/接电状态、后台程序、热降频都会影响；第 200 步脚本会打印实测外推的全程耗时 |
| `stage4` 提示权重结构不匹配 | 训练和生成必须用同一个 `PRESET`，否则 hidden_size/layers 对不上 |
| 不要合盖 | macOS 合盖休眠会中断训练；长时间跑请接电并关闭休眠 |

---

## 八、这套材料依据的实测

- 网页原版 `MiniMindForCausalLM` 在 MPS + BF16 下初始 loss ≈ 8.84，与网页的 ln(6400)≈8.76 对齐
- 原版 `AutoTokenizer` / `PretrainDataset` / `model.generate()` 均在本机验证可运行
- 原版代码路径实测吞吐（含实时分词）：`384/6 层 seq=256 b=16` 约 0.325 s/step；`256/4 层 seq=128 b=32` 约 0.148 s/step
- 微缩端到端实测：5M 模型 / 4 万条真实语料 / 3 epoch / 471 次参数更新，7.5 分钟，loss 从高初始值降到 5.39，续写出现主题词但重复
- 真实语料抽样（1565 条）：token 长度 p50=179、p90=379、均值 209.5
- **standard 档全链路实跑校准（2026-11-09）**：`stage0` 分 8 段下载 160 MB 全部成功，18.8 万条采样池抽出 12 万条（102.5 MB），语料 token p50=211、均值 237，seq=256 保留 78.8% 有效 token；`stage3` 用真实数据跑 300 步，**0.32 s/step、loss 8.12 → 7.43 → 7.37**，checkpoint 保存并被 `stage4` 成功加载生成。按 15,000 步外推全程约 1.3 小时（峰值），含降频约 1.7 小时，与上表一致
- 训练 300 步时的续写以「的、了、和、有」等高频虚词为主——模型先学会 unigram 分布，再学实词搭配，最后才成句。这本身就是很好的教学观察点

**2026-11-09 完整实跑更新**：stage3 的 15,000 步已全部跑完，耗时 5,175 秒（**86.3 分钟**，平均 0.345 s/step），loss 8.110 → **3.029**、降幅 5.08，明显优于早期「4.5–5.5」的保守外推。`stage4_generate.py` 实测中文续写通顺（如「人工智能是 → 计算机科学中最重要的领域之一…」）；英文 prompt 输出破碎，是 13.52M 小模型 + 中文为主语料的正常现象。上面表格该档的预期已按实测修正。
stage0 仍用分段 HTTP Range 采样语料，是 1.2 GB 全量的一个子集，分布比全量窄，但对教学环节无影响。

---

## 九、进阶：跟着网页走到 SFT，让模型会说话

### 为什么必须加这一段

网页这一章的产物**只会续写，不会对话**。网页第四阶段源码注释原文：

> 预训练模型只能做文本续写，SFT 之后才能做指令跟随对话

「会说话」是下一章 [工程实训：SFT 模型对话实验](https://ai.icyfenix.cn/language-models/pretraining/llm-sft-experiment.html) 的事：
把语料换成 ChatML 格式的问答对，**只让 assistant 回答部分参与 loss**，模型才学会「用户提问 → 我给出回答」。
网页自己的结论也写明：「SFT 训练使模型的行为发生质变，赋予了模型对话能力，但并未显著增加知识量或推理能力」。

所以前五步（stage0~4）原样不动，下面接上网页下一章的降配版 **stage5 → stage6 → stage7**。

### 一个绕不开的桥：起点权重

网页 SFT 章的起点固定是 `pretrain_768.pth`（768/8 层，即第 1 章的产物）。本机 stage3 训的是
384/6 层降配模型，**结构接不上**；而且它相对欠训练（实测 loss 3.03 vs 网页 1.85），硬接 SFT 也只能学会对话格式、
救不回语言流畅度。此前实测也确认：把官方 SFT 权重加载进本材料的 `MiniMindConfig(768,8)`
是 `missing=0 / unexpected=0`，与网页模型类完全一致。

因此默认 `SFT_BASE=official`：**下载作者官方已训好的 `pretrain_768.pth`（131MB）**——它正是
网页第 1 章「应该产出」的那个文件，结构、路径、后续流程与网页 100% 对齐。相当于把第 1 章
的产物补齐，再从第 2 章继续照着网页往下走。

想做对照实验（看欠训练 base 能不能学会格式）：`SFT_BASE=local`，改用你自己 stage3 的权重。

### 网页 SFT 章 → 本地映射

| 网页 SFT 章 | 网页动作 | 本机 |
|---|---|---|
| 实验准备 | `dmla data` 下 MiniMind SFT 语料（5 万条 / 90MB） | `stage5_sft_data.py [2/4]` 直链下载 |
| 实验准备 | 检查第 1 章产物 `pretrain_768.pth` | `stage5_sft_data.py [2/4]` 下载官方权重 |
| SFT 数据集 | `SFTDataset` / ChatML / labels 掩码 | 读 `shared/sft_dataset.py`，`stage5 [4/4]` 展示一条样本 |
| 训练代码块 | 加载 pretrain 权重 → SFT → 保存 `full_sft_768.pth` | `stage6_sft_train.py` |
| 推理代码块 | `apply_chat_template` + `chat()` 对话服务 | `stage7_sft_chat.py` |
| 实验结论 | 「从续写到对话的转变」 | `stage7` 的 [A] 续写 / [B] 对话 / [C] 官方权重 三组对照 |

### 逐步操作

> 下面三步**由你自己执行**。材料只做过 20 步冒烟校准与单步耗时实测，
> 正式 SFT 训练没有替你跑；数据和官方权重已全部备好，直接开跑即可。

```bash
cd dmla-mps-lab

# 5 · 实验准备 + SFT 数据集（约 1 分钟，下载 86MB 语料 + 131MB 权重）
.venv/bin/python stage5_sft_data.py

# 6 · SFT 训练（约 30 分钟；先冒烟可选）
LAB_SFT_LIMIT_STEPS=20 .venv/bin/python stage6_sft_train.py    # 可选冒烟
.venv/bin/python stage6_sft_train.py                            # 正式

# 7 · 对话（约 1 分钟）
.venv/bin/python stage7_sft_chat.py

# 对照实验：用自己 stage3 训的降配 base 再走一遍
SFT_BASE=local .venv/bin/python stage5_sft_data.py
SFT_BASE=local .venv/bin/python stage6_sft_train.py
SFT_BASE=local .venv/bin/python stage7_sft_chat.py
```

**第 5 步看什么**：ChatML 原文（`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...`）、
`labels` 里 `-100` 的比例、只有 assistant 回答参与 loss。这是本章与预训练最本质的差别。
产物：`data/sft_t2t_tiny.jsonl`(86M)、`data/sft_subset.jsonl`(1.5 万条 / 18.8M)、
`data/official_pretrain_768.pth`(131M)。

**第 6 步看什么**：`结构匹配 missing=0 unexpected=0`、初始 loss 约 1.9（远低于预训练的
ln(6400)≈8.76，因为起点已是会说话的模型，且只有回答部分算 loss）、每 50 步一行 loss、
第 200 步的耗时校准、warmup 10% + 余弦衰减的 lr 变化。产物：`sft_checkpoints/full_sft_official.pth`、
`sft_loss_log.tsv`。

**第 7 步看什么**：[A] 同一个起点权重、SFT 之前用第 1 章的 BOS+文本方式提问 → 只会顺着续写；
[B] SFT 之后用 ChatML 提问 → 识别 user/assistant 结构并给出助手式回答；
[C] 自动加载已备好的官方 `full_sft_768.pth`，作为网页标准答案对照。

### 与网页 SFT 章的差异

| 项 | 网页 | 本机 | 原因 |
|---|---|---|---|
| 起点权重 | 自己第 1 章训的 `pretrain_768.pth` | 官方同款 `pretrain_768.pth` | 降配 base 结构不匹配且欠训练 |
| 模型 | 768 / 8 层（64M） | **768 / 8 层（不变）** | 与网页完全对齐 |
| `max_seq_len` | 768 | 512 | 语料 p50=468；512 仍可完整保留 48.8% 样本 |
| 训练数据 | 5 万条 × 2 epoch | 筛过的 1.5 万条 × 1 epoch | 控制在 30 分钟量级 |
| batch / accum | 16 / 2 | 8 / 4 | **等效批仍 32** |
| `learning_rate` | 5e-5 | 5e-5（不变） | 低一个数量级，避免灾难性遗忘 |
| warmup + 余弦 | 10% + 余弦 | 不变 | 网页原函数 |
| device / DataLoader | cuda / 2 workers | mps / 0 workers | macOS spawn 易失败 |
| checkpoint | FP16 `.half()` | FP32 | MPS 部分算子不支持 FP16 权重 |

**没改的**：ChatML 模板、`apply_chat_template`、`generate_labels` 掩码逻辑、
`pre_processing_chat`、AdamW、梯度裁剪 1.0、梯度累积、周期保存、seed=42、
以及 stage7 的全部采样参数。

`shared/sft_dataset.py` 只有一处适配：网页用 `datasets.load_dataset` 读 JSONL，
本机 `.venv` 没装 `datasets`（会连带 pyarrow 等重依赖），换成纯 Python 逐行 `json.loads`；
算法部分与网页逐行一致。

### 实测依据

- SFT 语料 `sft_t2t_tiny.jsonl` 共 5 万条 / 86MB，token 长度 **p50=468 / p90=639 / mean=456**。
  `seq=256` 只能完整保留 8.7%（几乎全是极短拒答），`seq=512` 可留 48.8%（24,389 条）——这是把
  seq 从 768 降到 512 而不是 256 的依据。
- M5 / MPS / BF16 实测 768/8 模型单步（含 AdamW）：`seq=512 b8` **0.915 s/step** → 1,875 步约 29 分钟。
- 全链路冒烟（20 步 / 0.3 分钟）通过：官方权重 `missing=0 unexpected=0`、FP16 自动转 FP32、
  SFT 初始 loss 1.906、stage7 的 [A] 续写 / [B] 对话均生成成功。
- 官方 `full_sft_768.pth` 单独实测：生成 64 token 仅 0.6–1.7 秒，能说完整句子、按对话格式回答，
  但会胡说（「1+1」答 3）。**能对话 ≠ 事实正确**，这正是 64M 模型的真实上限。
- **完整 1,875 步正式 SFT 训练未跑完**（校准只跑到 800 步便停止，loss 1.86 → 1.75）。
  因此上面 30 分钟是前 200 步实测外推，「训练完的对话质量」**未经实测**，留给你自己跑。
  校准产生的半成品 checkpoint / loss 日志已移出项目，当前 `sft_checkpoints/` 为空。

### 已为你备好的文件（不用再下载）

`data/` 下四个文件已就绪，stage5 重跑会走缓存、不会重复下载：

| 文件 | 大小 | 用途 |
|---|---:|---|
| `sft_t2t_tiny.jsonl` | 86 MB | 网页 SFT 语料，5 万条 |
| `sft_subset.jsonl` | 19 MB | 筛好的 1.5 万条短对话训练集 |
| `official_pretrain_768.pth` | 131 MB | SFT 起点权重（网页第 1 章产物） |
| `official_full_sft_768.pth` | 131 MB | 网页标准答案，stage7 的 [C] 对照会自动加载 |

### 验收清单（SFT 部分）

- [ ] 能说清「预训练每个位置都算 loss，SFT 只有 assistant 回答算 loss」在代码里是哪一行
- [ ] 能说清 ChatML 模板为什么必须手动设置（tokenizer 未内置）
- [ ] `stage6` 打印的等效批是 32、初始 loss 明显低于 8.76
- [ ] `stage7` 的 [A] 与 [B] 行为差异肉眼可辨
- [ ] 能解释「回答仍有事实错误」= 64M 容量 + SFT 不增加知识，而不是训练失败
- [ ] 对照跑一次 `SFT_BASE=local`，能说清欠训练 base 为什么学不会说话

### 想要一个真正好用的助手

这套材料的终点是**教学用的 64M 小模型**，语言流畅、会对话，但知识与推理有限。
若要日常可用，直接在本机 MPS 上跑现成小模型（如 Qwen3.5-0.8B-Instruct），
那属于网页「模型服务化」章节，不在本实验范围内。

---

## 十、随时试玩：输入一句话，看模型接什么

不想跑完整 stage4，只想随手敲几个句子看模型反应，用这个：

```bash
.venv/bin/python play_pretrain.py                 # 进入交互
.venv/bin/python play_pretrain.py "人工智能是"     # 只跑一次就退出
```

交互里直接输入文字即可，模型会顺着往下续写。可调采样的命令：

| 命令 | 作用 |
|---|---|
| `:len 128` | 改每次生成的 token 数（默认 64） |
| `:temp 0.7` | 改温度，越低越保守、越高越乱（默认 0.85） |
| `:top 0.9` | 改 top_p（默认 0.85） |
| `:params` / `:help` | 查看当前参数 / 帮助 |
| `:q` / `exit` | 退出；Ctrl-C / Ctrl-D 也可以 |

采样参数与网页第四阶段一致（`top_k=50`、`repetition_penalty=1.2`）。实测输出：

```
> 人工智能是
  怎样的？它是一种模拟人类智能的技术，通过对大量的数据和模式进行分析、学习、
  推理和决策。这种技术包括深度学习、自然语言处理、计算机视觉等技术……
```

**务必记住这是 base 模型**：每轮独立、不记得上一句、只会续写不会回答问题。
输入「你好，你是谁」得到的是顺着这句话编下去的文本，不是回答——这不是 bug，
对话能力要 `stage5 → stage6 → stage7`（SFT）才有。英文 prompt 输出破碎也属正常，
因为语料以中文为主、词表只有 6400。

脚本只读 `checkpoints/pretrain_<PRESET>_final.pth`，不训练、不写任何文件；
换档位用 `PRESET=quick .venv/bin/python play_pretrain.py`（需有对应档位的权重）。
