# 与网页实验原版的完整差异说明

> 配套文章：[Transformer 模型预训练实验](https://ai.icyfenix.cn/language-models/architecture-basics/llm-pretrain-experiment.html)（第 1 章）、[工程实训：SFT 模型对话实验](https://ai.icyfenix.cn/language-models/pretraining/llm-sft-experiment.html)（第 2 章）
>
> 操作手册见 [GUIDE.md](GUIDE.md)，项目定位见 [README.md](README.md)，署名与许可证见 [NOTICE.md](NOTICE.md)。

---

## 一、结论先行

**本项目是「教学降配 + 多平台适配」的补充方案，不是网页实验的复现。**

| | 网页原版 | 本项目 |
|---|---|---|
| 要证明什么 | 一次接近真实规模的语言模型预训练与指令微调 | 在算力不足的机器上，把**同一条学习链路**亲手走通 |
| 成功标准 | loss 降到约 1.85，模型具备基本语言与对话能力 | 跑通全流程、看懂 loss 曲线、亲手观察「续写 → 对话」的转变 |
| 适用读者 | 有 Docker + NVIDIA GPU 的环境 | 没有独显、显存不够、或只有 Apple Silicon / 纯 CPU 的读者 |

**相同的是流程与算法，不同的是规模、设备、数据获取方式和最终指标。** 模型定义、数据集类、损失函数、优化器、调度器、ChatML 模板都沿用网页原版；改的只有「跑在什么上面」和「跑多大」。

---

## 二、运行环境对照

| 维度 | 网页原版 | 本项目 |
|---|---|---|
| 运行方式 | 网页里的 **Run 按钮** → 云端 Serverless 沙箱，或自建 DMLA Docker 沙箱 | 本地终端**逐个执行 Python 脚本** |
| 必备软件 | Docker、NVIDIA Container Toolkit、CUDA 13、Git LFS、NodeJS（DMLA CLI） | Python ≥ 3.11、`uv`、Git；**不需要 Docker / CUDA / Git LFS / NodeJS** |
| 计算后端 | CUDA（第 1 章 6 个代码块中 5 个标了 `data-gpu-only="true"`） | 自动检测 **CUDA > MPS > CPU**，可用 `LAB_DEVICE` 强制指定 |
| GPU 是否必需 | 是；云端 FC 端点因费用原因**未开放 GPU**，训练章需自行部署沙箱 | 否；纯 CPU 也能跑完（只是慢一个数量级） |
| 操作系统 | Linux + N 卡（DMLA 镜像为 amd64） | macOS / Linux / Windows（含 WSL2） |
| 数据获取 | `dmla data` 全量下载预训练语料 1.2 GB + SFT 语料 90 MB（Git LFS） | Python 标准库 **HTTP Range 分段采样**：预训练约 160 MB、SFT 86 MB、官方权重按需 131 MB |
| 下载依赖 | DMLA CLI + Git LFS | 无外部命令依赖（`shared/download.py`，urllib 实现） |
| 混合精度 | CUDA BF16 AMP | 后端支持 BF16 时启用（MPS / 支持 BF16 的 N 卡），CPU 与老卡保持 FP32 |
| 注意力算子 | FlashAttention（CUDA 路径） | PyTorch `F.scaled_dot_product_attention`，三种后端均有原生实现 |
| DataLoader | `pin_memory=True, num_workers=2` | `False, 0`（单进程跨平台最稳） |
| checkpoint 精度 | FP16（`.half().cpu()`） | FP32（跨平台互读，避免 MPS/CPU 的 FP16 算子限制） |
| 进度上报 | 沙箱注入的 `ProgressReporter`、`DATA_DIR` | 普通 stdout + `loss_log.tsv` |

> 为什么不做 CUDA 专属优化：网页的 CUDA 加速路径依赖 FlashAttention、Tensor Core 与融合 kernel，Apple Silicon 没有对应实现；本项目改用 PyTorch 的通用 SDPA，换来的是**一套代码三种后端**，代价是 MPS/CPU 上单步更慢。

---

## 三、模型与超参逐项对照

### 3.1 预训练（第 1 章）

| 项 | 网页原版 | 本项目 standard 档 | 说明 |
|---|---|---|---|
| `hidden_size` | 768 | **384** | 宽度减半；FFN 由配置类公式自动从 2432 → 1216 |
| `num_hidden_layers` | 8 | **6** | |
| 参数量 | 63.91M | **13.52M** | 约 **4.7×** 差距 |
| GQA 头数 | 8 Q / 4 KV | 同（按公式推导） | **未改** |
| `vocab_size` / 权重共享 | 6400 / True | 同 | **未改** |
| `max_seq_len` | 512 | **256** | 依据真实语料长度分布，见第七节 |
| `batch_size` | 32 | **16** | |
| `accumulation_steps` | 8 | **16** | |
| **等效批大小** | **256** | **256** | **刻意保持不变**，只改变梯度累积频率 |
| 训练语料 | 约 127 万条 × 2 epoch | **12 万条 × 2 epoch** | 约 10.6× 差距 |
| `learning_rate` | 5e-4 | **1e-3** | 小模型收敛更快；`faithful` 档仍用 5e-4 |
| 优化器 / 调度 | AdamW + 余弦衰减 | 同 | **未改**（连 `get_lr()` 的实现都没动） |
| 混合精度 / 梯度裁剪 | BF16 AMP / 1.0 | 同（按后端启用） | **未改** |
| 参数更新次数 | 约 9,900 | 约 937 | 约 10.6× 差距 |
| 有效训练 token | 约 6.5 亿 | 约 5,500 万 | 按有效 token 估算约 **11.9×** 差距 |
| 设备 / 参考耗时 | RTX 5080 ≈ **2.5 小时** | Apple M5 ≈ **86 分钟** | 数字不能直接比：模型小 4.7×、token 少 11.9× |
| 最终 loss | 约 **1.85** | 实测 **3.029** | 不追求复现 |
| 随机基线 | `ln(6400) ≈ 8.76` | 同 | 本项目实测从 8.110 起降 |

### 3.2 SFT（第 2 章）

| 项 | 网页原版 | 本项目 | 说明 |
|---|---|---|---|
| 起点权重 | 第 1 章产物 `pretrain_768.pth`（768 / 8 层） | **官方同结构权重** | 降配版第 1 章是 384 / 6 层，结构接不上；用作者发布的 `pretrain_768.pth` 补齐「网页第 1 章应该产出的文件」 |
| 对话语料 | 约 5 万条 / 90 MB | **1.5 万条短对话** | 按 token 长度筛出能完整放进 seq 的样本 |
| `max_seq_len` | 768 | **512** | 实测 p50=468 / p90=639，seq=256 只剩 8.7% 样本且几乎都是极短拒答，故定 512 |
| `batch_size` / `accumulation` | 16 / 2 | **8 / 4** | 等效批 **32 不变** |
| `num_epochs` | 2 | **1** | |
| `learning_rate` / warmup | 5e-5 / 10% | 同 | **未改** |
| ChatML 模板 | 有 | 逐行保留 | **未改** |
| labels 掩码 | 只监督 assistant 区间 | 逐行保留 | **未改**（本章最核心的教学点） |
| 参考耗时 | RTX 5080 ≈ 15 分钟 | M5 约 30 分钟 | 前 200 步外推；**完整 1,875 步训练未跑完，最终对话质量未实测** |

> 可用 `SFT_BASE=local` 改用自己 stage3 训出的 384 / 6 权重做对照，观察欠训练 base 能否学会对话格式——这是网页没有的对照实验。

---

## 四、逐阶段操作对照

### 4.1 第 1 章（预训练）

网页是 **1 条终端命令 + 6 个可点击代码块**；本项目是 5 个脚本 + 1 个可选交互工具。

| # | 网页 | 本项目 | 对应关系 |
|---|---|---|---|
| 0 | `dmla data`（终端，下载全量数据） | `stage0_setup.py` | 数据准备 + 文件检查，附带 token 长度分布统计 |
| ① | Run ① 环境检查 / 分词器 | `stage0_setup.py`、`stage1_tokenizer.py` | 环境与依赖自检 |
| ② | Run ② 分词器演示 | `stage1_tokenizer.py` | BOS/EOS/PAD、压缩率 |
| ③ | Run ③ `PretrainDataset` 定义 | `stage1_tokenizer.py` + `shared/pretrain_dataset.py` | 额外讲清 `-100` 与「右移其实在模型 forward 里」 |
| ④ | Run ④ `MiniMindConfig` / 模型定义 | `stage2_model.py` + `shared/mini_mind_config.py` | 额外补参数量手算与显存重算 |
| ⑤ | Run ⑤ 预训练（`data-timeout="unlimited"`） | `stage3_train.py` | 训练循环逐行对应 |
| ⑥ | Run ⑥ 推理续写 | `stage4_generate.py` | 保留网页 4 个 prompt 与全部采样参数 |
| — | 无 | `play_pretrain.py` | 额外：输入一句话看模型接什么 |

### 4.2 第 2 章（SFT）

| # | 网页 | 本项目 |
|---|---|---|
| 准备 | `dmla data` 取 SFT 语料 + 第 1 章产物 | `stage5_sft_data.py`（下载语料与官方起点权重、构建短对话子集、展示 ChatML 与掩码） |
| 训练 | SFT 训练代码块 | `stage6_sft_train.py` |
| 对话服务 | 推理 / 对话代码块 | `stage7_sft_chat.py`（额外加 A/B/C 三组教学对照） |

**不能覆盖的字面操作**：网页 Run 按钮、DMLA CLI 全量数据下载、原版 64M 全量训练、`loss=1.85` 这个数值结果。这些是网页 GPU 沙箱能力，不是本项目目标。

---

## 五、代码层面：改了什么 / 没改什么

### 5.1 逐文件来源与改动

| 文件 | 来源 | 改动 |
|---|---|---|
| `shared/mini_mind_config.py` | DMLA 第 2 阶段代码块 | 取纯模型定义版，去掉网页代码块末尾的演示 `print`；**模型结构零改动** |
| `shared/pretrain_dataset.py` | DMLA 第 1 阶段代码块 | **原样保留** |
| `shared/sft_dataset.py` | DMLA SFT 章 `sftdataset.py` | 仅把 `datasets.load_dataset` 换成纯 Python 逐行 `json.loads`，避免引入 `datasets` / `pyarrow` 重依赖；ChatML、`generate_labels()` 掩码、`pre_processing_chat()` 算法逐行保留 |
| `stage3_train.py` | 网页第 3 阶段代码块 | 去掉沙箱专属 `ProgressReporter` / `DATA_DIR` 注入；设备改自动检测；DataLoader 单进程；checkpoint FP32；增加 loss 曲线、有限值断言、耗时校准 |
| `stage6_sft_train.py` | 网页 SFT 训练代码块 | 同上；保留 warmup + 余弦、lr=5e-5、ChatML 掩码 |
| `stage1/2/4/7` | 网页各阶段演示 / 推理代码块 | 设备与路径本地化；**保留网页的 prompt、ChatML 模板与全部采样参数** |
| `config.py`、`stage0/5`、`play_pretrain.py`、`shared/download.py`、`setup.sh`、`setup.ps1`、全部文档 | **本项目原创** | 配置中心、数据准备、交互试玩、标准库下载器、多平台安装脚本 |

### 5.2 明确没改的算法

- next-token prediction 的右移实现：`logits[..., :-1, :]` 对 `labels[..., 1:]`，`ignore_index=-100`
- `MiniMindConfig` 的结构公式（GQA / RoPE / SwiGLU / 权重共享）
- AdamW、`get_lr()` 余弦调度、warmup、梯度累积的触发条件、梯度裁剪 1.0、周期保存、seed=42
- ChatML 模板与「只监督 assistant 回答」的 labels 掩码
- `generate()` 的采样参数：`temperature=0.85`、`top_p=0.85`、`top_k=50`、`repetition_penalty=1.2`

### 5.3 本项目额外增加的教学内容

token 长度分布与 padding 浪费的量化、参数量手算、显存重算、loss 曲线落盘与纯文本 sparkline、训练前后 + 随机初始化对照、SFT 前/后/官方权重三组对照、交互续写 REPL、三档预设、短程冒烟（`LAB_LIMIT_STEPS`）。

---

## 六、能得到什么 / 得不到什么

### 能得到

- 完整链路：数据 → 分词 → Dataset → 模型 → 训练循环 → loss 曲线 → checkpoint → 推理 → SFT → 对话
- 亲手观察：loss 从 ≈8.76 下降、初期陡降后期趋缓与余弦 LR 的关系、欠训练导致的重复生成
- 工程手感：混合精度开关、梯度累积（等效批不变）、梯度裁剪、checkpoint 读写、权重结构与档位匹配
- 核心机制：SFT 的 ChatML + labels 掩码如何把「续写模型」变成「对话模型」
- 一组可复现的实测数字与曲线样例（`examples/loss_log_standard_sample.tsv`）

### 得不到（请明确预期）

- 网页的 `loss ≈ 1.85`、64M 全量指标、2.5 小时 / RTX 5080 的对应关系
- 一个日常可用的聊天助手，或事实可靠、推理正确的模型
- 英文能力（语料以中文为主、词表仅 6400，英文 prompt 输出破碎）
- CUDA / FlashAttention / Tensor Core / 多卡分布式训练的真实经验
- 网页 GPU 沙箱的 Run 按钮与 `dmla data` 全量数据

> 网页源码注释与本章结论都写明：**预训练模型只会文本续写，SFT 之后才具备指令跟随**；而 SFT 赋予的是「对话行为」，不显著增加知识与推理能力。官方 768/8 权重在本项目实测中会把「1+1」答成 3，这是 64M 级模型的正常表现，不是 bug。

---

## 七、为什么这样降配

1. **序列长度**：真实语料 token 长度抽样 p50≈211、p90≈378、p95≈450、mean≈237、max≈1739。`seq_len=512` 时约 **59% 的计算花在 padding 上**；`seq_len=256` 只损失少量有效 token，却把注意力计算量减半。这是「缩短序列几乎不损教学效果」的依据。
2. **等效批保持 256**：把 `batch 32×accum 8` 改成 `16×16`，改的是梯度累积频率，优化目标与更新粒度不变，因此 loss 曲线形状与网页仍可比。
3. **模型结构不改公式**：只传 `hidden_size` 与层数，FFN 维度、head_dim、GQA 头数由 `MiniMindConfig` 自动推导，保证「参数量手算」这一步仍能与网页表格逐项核对。
4. **三档预设**：`quick`（4.98M / seq128 / 8 万条，12~18 分钟跑通链路）、`standard`（13.52M / seq256 / 12 万条，实测 86 分钟）、`faithful`（63.91M / seq256 / 10 万条 × 1 epoch，保留原模型规模便于核对显存表）。三档等效批恒为 256。
5. **SFT 起点用官方权重**：不是偷懒，是为了让第 2 章在结构上与网页 100% 对齐；想看降配影响，用 `SFT_BASE=local` 做对照。

---

## 八、实测记录对照

| 项目 | 网页原版 | 本项目（Apple M5 / 16GB） |
|---|---|---|
| 设备 | RTX 5080（16GB GDDR7） | Apple M5（8 核 GPU / 16GB 统一内存） |
| 预训练 | 约 2.5 小时，loss → 1.85 | 15,000 步 / **86.3 分钟** / 平均 0.345 s/step，loss **8.110 → 3.029** |
| checkpoint | — | 10 个，各约 61 MB，FP32，`missing=0 / unexpected=0` |
| 续写 | — | 中文通顺、有主题词，但内容会胡说、重复；英文 prompt 破碎 |
| 权重健康 | — | 无 NaN/Inf；用 `MiniMindConfig(384,6)` 加载完全匹配 |
| SFT | RTX 5080 约 15 分钟 | 全链路 20 步冒烟通过（初始 loss 1.906）；**完整训练未跑完** |

**两个耗时数字不构成性能对比**：本项目模型小 4.7×、样本曝光少 10.6×、有效 token 少约 11.9×，属于不同规模的实验。M5 上跑网页原版规模曾外推需要 66~105 小时以上，这正是需要降配的原因。

---

## 九、许可证与使用边界

上游《设计机器学习应用系统》采用 **CC BY-NC-SA 4.0**（署名、非商业、相同方式共享），其 README 明确要求衍生作品以同一许可证发布。因此本项目同样以 CC BY-NC-SA 4.0 发布，**不能改用 MIT / Apache 等宽松许可证**，也不得用于商业目的或商业环境内的培训。逐文件引用与改编清单见 [NOTICE.md](NOTICE.md)。

本项目不是 DMLA 官方项目，与作者无关联；内容若有出入，以原网页为准。

---

## 十、常见问题

| 问题 | 回答 |
|---|---|
| 能复现网页的 loss 1.85 吗？ | 不能。模型、数据量、token 量都差一个数量级，实测 3.029。想复现请用网页的 GPU 沙箱。 |
| 预训练完的模型能对话吗？ | 不能，只会续写。这是网页的设计，不是 bug；SFT 章才赋予对话行为。 |
| 我有 N 卡，还用这个项目吗？ | 现在脚本会自动选 CUDA，但降配仍在。想做原版实验请直接用网页沙箱；本项目适合快速过一遍流程或做对照。 |
| 为什么 SFT 用官方权重而不是自己训的？ | 网页 SFT 章固定以 768/8 的 `pretrain_768.pth` 为起点，降配版是 384/6，结构不兼容。`SFT_BASE=local` 可切回自己的权重。 |
| 数据是网页全量吗？ | 不是。用 HTTP Range 从语料不同位置分段采样约 160 MB，再随机抽 12 万条；SFT 按长度筛 1.5 万条。 |
| 没 GPU 能跑完吗？ | 能，链路完全一致，但耗时长一个数量级。建议先 `PRESET=quick`。 |
| 可以用于公司培训或产品吗？ | 不可以。许可证是非商业性的，详见第九节。 |
| 完整 SFT 结果怎么样？ | **尚未实测**。完整 1,875 步训练留给读者自己执行，README 与 GUIDE 均如实标注。 |
