# 署名与来源说明

本项目是《设计机器学习应用系统》(DMLA) 相关教学实验的 **Apple Silicon 适配 / 降配衍生作品**，仅供个人学习与非商业教学使用。

## 一、上游作品

| 项 | 内容 |
|---|---|
| 作品 | 设计机器学习应用系统（Designing Machine Learning Application, DMLA） |
| 作者 | 周志明（fenixsoft / icyfenix） |
| 网站 | https://ai.icyfenix.cn |
| 源码 | https://github.com/fenixsoft/dmla |
| 许可证 | **CC BY-NC-SA 4.0**（Attribution-NonCommercial-ShareAlike 4.0 International） |

上游 README 的 License 章节明确要求：

- **署名**：标明作者、提供许可证链接、说明是否修改
- **非商业性使用**：不得用于商业目的
- **相同方式共享**：基于本作品改编、重混的作品，必须以同一许可证分发

据此，**本项目整体以 CC BY-NC-SA 4.0 发布**（见 [LICENSE](LICENSE)），不使用 MIT / Apache 等宽松软件许可证。

涉及的网页章节：

- [Transformer 模型预训练实验](https://ai.icyfenix.cn/language-models/architecture-basics/llm-pretrain-experiment.html)
- [工程实训：SFT 模型对话实验](https://ai.icyfenix.cn/language-models/pretraining/llm-sft-experiment.html)

## 二、引用与改编清单

| 本仓库文件 | 来源 | 改动 |
|---|---|---|
| `shared/mini_mind_config.py` | DMLA `local-server/shared/llm/mini_mind_config.py`（对应网页第 2 阶段代码块） | 取纯模型定义版，去掉网页代码块末尾的演示 `print`；结构未改 |
| `shared/pretrain_dataset.py` | DMLA `local-server/shared/llm/pretrain_dataset.py`（对应网页第 1 阶段代码块） | 原样保留 |
| `shared/sft_dataset.py` | DMLA `local-server/shared/llm/sftdataset.py`（对应网页 SFT 章代码块） | 仅把 `datasets.load_dataset` 换成纯 Python 逐行 `json.loads`，避免引入 `datasets`/`pyarrow`；ChatML 模板、`generate_labels` 掩码、`pre_processing_chat` 等算法逐行保留 |
| `stage3_train.py` | 网页第 3 阶段训练代码块 | 保留 BF16 autocast、梯度累积、`get_lr` 余弦调度、梯度裁剪 1.0、周期 checkpoint、seed=42；去掉沙箱专属 `ProgressReporter` 与 `DATA_DIR` 注入，设备改 MPS，DataLoader 改单进程，checkpoint 存 FP32，增加 loss 日志与耗时校准 |
| `stage6_sft_train.py` | 网页 SFT 章训练代码块 | 同上；保留 warmup 10% + 余弦、lr=5e-5、ChatML 与 labels 掩码 |
| `stage0_setup.py` / `stage5_sft_data.py` | 网页「实验准备」的 `dmla data` 与文件检查 | 改为 ModelScope HTTP Range 直链下载，并按 token 长度筛出可完整放进序列的短对话 |
| `stage1/2/4/7` 等脚本 | 对应网页各阶段演示 / 推理代码块 | 设备改 MPS、路径本地化；保留网页的 prompt 与全部采样参数 |
| `GUIDE.md` 中引用的网页原文与代码块说明 | DMLA 网页正文 | 引用处均标注章节链接，仅作学习对照 |

`config.py`、`play_pretrain.py`、`setup.sh`、`README.md`、`examples/` 为本项目的组织性文件；训练流程与算法思路仍来自上述网页章节。

## 三、数据、分词器与模型权重来源

| 资源 | 上游 | 许可证 / 分发 |
|---|---|---|
| MiniMind 预训练语料、BPE 分词器 | [jingyaogong/minimind](https://github.com/jingyaogong/minimind) | Apache-2.0 |
| MiniMind SFT 对话语料 | 同上 | Apache-2.0 |
| `pretrain_768.pth` / `full_sft_768.pth` | DMLA 作者在 ModelScope 上发布的 `icyfenix/minimind` 镜像 | 随上游作品条款使用 |

ModelScope 镜像仓库：

- `icyfenix/Minimind_Pretrain`（语料 + 分词器）
- `icyfenix/Minimind_SFT`（SFT 语料）
- `icyfenix/minimind`（预训练 / SFT 权重）

**本仓库不包含以上任何二进制或数据文件**，也不包含任何训练产物。它们由 `stage0_setup.py` 与 `stage5_sft_data.py` 在首次运行时按需联网下载到 `data/`，权重由训练脚本生成到 `checkpoints/`、`sft_checkpoints/`；这些目录均在 `.gitignore` 中排除。使用者需自行遵守各上游资源的条款。

## 四、修改摘要（满足 BY 的「说明是否修改」要求）

1. 运行后端由 CUDA / Docker 沙箱改为 Apple Silicon 的 PyTorch **MPS**，混合精度用 MPS 的 BF16 autocast。
2. 数值降配：模型 13.52M（384 / 6 层）、序列 256、语料 12 万条；SFT 阶段序列 512、1.5 万条对话。等效批大小与网页保持一致（预训练 256、SFT 32）。
3. 权重保存由 FP16 改为 FP32（MPS 部分算子不支持 FP16 权重）；DataLoader 改为 `num_workers=0`。
4. SFT 起点改用上游发布的官方 `pretrain_768.pth`，因为降配版预训练权重结构（384 / 6 层）与网页 SFT 章（768 / 8 层）不兼容。
5. 增加档位预设、loss 曲线落盘、耗时校准、交互续写工具与中文学习手册。

## 五、使用限制

- **非商业性使用**：不得用于商业目的或商业环境内的培训/产品化。
- **相同方式共享**：分发或改编本项目时，必须继续使用 CC BY-NC-SA 4.0 并保留本署名文件。
- 本项目不是 DMLA 官方项目，与作者无关联；内容若有误，以原网页为准。
