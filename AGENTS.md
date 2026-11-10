# AGENTS.md

> 给**接手本项目的 AI 助手**的项目上下文。人类读者请看 [README.md](README.md)（定位）、[GUIDE.md](GUIDE.md)（逐步操作）、[COMPARISON.md](COMPARISON.md)（与网页原版的逐项差异）。
>
> 本文件是项目的一部分，同样以 **CC BY-NC-SA 4.0** 发布。增删文件后请同步更新 [NOTICE.md](NOTICE.md) 的原创文件清单与 README 的目录结构。

---

## 0. 三十秒版：先记住这五条

1. 本项目是《设计机器学习应用系统》(DMLA) 两章实验的**教学降配 + 多平台适配衍生作品**，**不是复现实验**。目标是让读者跑通链路、看懂 loss 曲线、观察「续写 → 对话」的转变，**不是训练出好模型**。
2. **不要「优化」成追求 `loss 1.85`、升级到 64M 全量或接入更大语料** —— 那会改变项目定位。
3. 许可证是 **CC BY-NC-SA 4.0**（上游 README 明确要求衍生作品同许可证 + 非商业）。**禁止改成 MIT / Apache，禁止商业用途**。
4. 仓库**不含**任何权重、语料、分词器、`.venv`、`checkpoints`。这些由脚本按需联网下载或训练生成，**禁止提交进仓库**。
5. **不要替用户跑正式训练**（用户明确要求）。技术验证请在 `/tmp` 隔离副本里做短程冒烟，绝不覆盖用户真实的 `loss_log.tsv` / `checkpoints/` / `data/`。

---

## 1. 文件地图

| 文件 | 职责 | 来源 |
|---|---|---|
| `config.py` | **唯一配置中心**：档位预设、路径、超参、设备检测、SFT 配置、ChatML 模板 | 本项目原创 |
| `stage0_setup.py` / `stage5_sft_data.py` | 数据准备：联网下载、构建子集、教学统计与掩码演示 | 本项目原创（替代网页 `dmla data`） |
| `stage1_tokenizer.py` / `stage2_model.py` | 分词器演示、参数量手算、显存重算 | 本项目原创（教学脚本） |
| `stage3_train.py` / `stage6_sft_train.py` | 预训练 / SFT **训练循环** | 网页代码块改编，算法逐行保留 |
| `stage4_generate.py` / `stage7_sft_chat.py` | 续写推理 / ChatML 对话 + 教学对照组 | 网页代码块改编 |
| `play_pretrain.py` / `play_sft.py` | 交互续写 REPL / 交互对话 + 权重对比 + 批量测试，均只读权重 | 本项目原创 |
| `shared/mini_mind_config.py` | `MiniMindConfig` / `MiniMindForCausalLM` 模型定义 | **上游原版**，仅去演示 print |
| `shared/pretrain_dataset.py` | 预训练 Dataset | **上游原版**，原样保留 |
| `shared/sft_dataset.py` | SFT Dataset + ChatML + labels 掩码 | 上游改编，**仅替换 IO**（`datasets.load_dataset` → 纯 Python `json.loads`） |
| `shared/download.py` | 标准库 urllib + HTTP Range 下载器 | 本项目原创 |
| `setup.sh` / `setup.ps1` | macOS·Linux / Windows 环境安装 | 本项目原创 |
| `requirements.txt` | `torch==2.15.0`、`transformers==5.22.1`、`tokenizers>=0.22` | 本项目原创 |
| `README.md` / `GUIDE.md` / `COMPARISON.md` / `NOTICE.md` / `LICENSE` / `AGENTS.md` | 定位 / 手册 / 差异 / 署名 / 许可证 / 接手说明 | 本项目原创 |
| `examples/loss_log_standard_sample.tsv` | 真实 150 点 loss 曲线样例 | 本项目原创 |

**不要改的算法**：模型结构与 `forward` 里的 next-token 右移、Dataset 类、ChatML 模板与 `generate_labels()` 掩码、AdamW、`get_lr()` 余弦调度、梯度累积触发条件、梯度裁剪 1.0、seed=42、`generate()` 采样参数。除非有明确理由，且改动必须同步更新 `COMPARISON.md` 的「改了 / 没改」清单。

---

## 2. 两个目录，必须保持同步

项目通常同时存在两份：

| | 发布仓库（本文件所在处） | 本地运行目录 |
|---|---|---|
| 内容 | 只有代码 + 文档，可 clone | 另有 `.venv/`、`data/`、`checkpoints/`、`sft_checkpoints/`、真实 loss 日志 |
| 用途 | 发布、给别人用 | 用户实际跑实验 |
| 是否为 git 仓库 | 是（`main`，remote `github.com/thecoolboyhan/dmla-mps-lab.git`） | 否 |

**规则**：修改任何共有代码文件（`config.py`、`stage0~7`、`play_pretrain.py`、`play_sft.py`、`shared/*.py`、`setup.sh`、`setup.ps1`、`requirements.txt`、`GUIDE.md`、`AGENTS.md`）后，**两份都要同步，逐字节一致**。尤其注意 `stage0`/`stage5` 依赖 `shared/download.py`，新增模块忘记同步会直接 import 失败。

提交公开发布仓库时用**仓库级** git 身份（`thecoolboyhan <thecoolboyhan@users.noreply.github.com>`），不要用全局身份——全局配置带有不应出现在公开 commit 里的邮箱。

---

## 3. 配置与开关（全部在 `config.py`，不要散落到脚本里）

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `PRESET` | `standard` | 档位：`quick`(4.98M/128) / `standard`(13.52M/256) / `faithful`(63.91M/256) |
| `LAB_SUBSET_SAMPLES` | 档位预设 | 覆盖训练样本数 |
| `LAB_DOWNLOAD_MB` | 档位预设 | 覆盖分段下载量 |
| `LAB_LIMIT_STEPS` | `0` | 只跑 N 步（冒烟 / 校准），**不影响余弦 LR 分母** |
| `LAB_DEVICE` | 自动 | 强制 `cuda` / `mps` / `cpu`；不可用时主动报错 |
| `LAB_BF16` | 按后端 | `0` 关 / `1` 开，用于 FP32 对照 |
| `SFT_BASE` | `official` | `official` 用官方 768/8 权重；`local` 用自己 stage3 的 384/6 权重 |
| `LAB_SFT_SAMPLES` | `15000` | SFT 子集大小 |
| `LAB_SFT_LIMIT_STEPS` | `0` | SFT 短程冒烟 |

**设备优先级**：`cuda > mps > cpu`。BF16 只在 MPS 或 `torch.cuda.is_bf16_supported()` 为真时启用；CPU 与老卡保持 FP32（`nullcontext()`）。

**不变量**：三档等效批恒为 **256**（`batch_size × accumulation_steps`）；SFT 等效批恒为 **32**。改数值时保持这个乘积，否则 loss 曲线与网页失去可比性。

`PY_CMD` 会根据 `os.name` 输出 `.venv/bin/python` 或 `.venv\Scripts\python.exe`，脚本里的「下一步」提示都用它；**不要在脚本里硬编码 POSIX 路径**。

---

## 4. 运行顺序（不要跳步）

用 `<venv-python>` 代指平台解释器（macOS/Linux `.venv/bin/python`，Windows `.venv\Scripts\python.exe`）：

```bash
bash setup.sh                    # Windows: powershell -ExecutionPolicy Bypass -File .\setup.ps1
<venv-python> stage0_setup.py    # 首次会联网下载约 160 MB
<venv-python> stage1_tokenizer.py
<venv-python> stage2_model.py
<venv-python> stage3_train.py    # 长任务；M5 standard 档 86 分钟
<venv-python> stage4_generate.py
<venv-python> play_pretrain.py   # 可选，只读权重
<venv-python> stage5_sft_data.py # 首次会联网下载语料 + 官方权重
<venv-python> stage6_sft_train.py # 1,875 步；M5 实测 28.1 分钟
<venv-python> stage7_sft_chat.py
<venv-python> play_sft.py        # 可选，自定义测试对话效果
```

`stage3` 需要 `stage0` 产物；`stage6` 需要 `stage5` 产物；`stage7` 需要 `stage6` 产物。切 `PRESET` 后各阶段必须用**同一档**，否则权重结构不匹配。

---

## 5. 已知事实（不要重新推测，也不要擅自改文档里的数字）

- Apple M5 / 16GB 实测：standard 档 **15,000 步 / 86.3 分钟 / 平均 0.345 s/step**，loss **8.110 → 3.029**，10 个 FP32 checkpoint（各约 61 MB），`missing=0 unexpected=0`。
- 随机基线 `ln(6400) ≈ 8.76`。中文续写通顺但会胡说、重复；英文 prompt 破碎（中文语料 + 6400 小词表）。
- **2026-11-10 完整 SFT 已实跑**：1,875 步 / 1,683.5 秒（28.1 分钟），loss **1.860 → 1.647**，38 个记录点，3 个 FP32 checkpoint（各约 263 MB）。
- SFT 后对话实测：会按 ChatML 格式回答，但「1+1」答 2.5、自我介绍自称 Qwen；官方 `full_sft_768.pth` 同量级胡说。**格式学会 ≠ 事实正确**，与网页结论一致。
- `play_sft.py` 是只读的自定义测试工具：`--list` / `--compare` / `--ckpt` / `--batch` / `--greedy` 均已实跑验证；不要把它改成训练脚本。
- **CUDA 分支只做了逻辑测试**，本机没有 N 卡，真机完整训练未实测。不要声称「已在 NVIDIA GPU 上验证」。
- next-token 的**右移发生在模型 `forward`**：`logits[..., :-1, :]` 对 `labels[..., 1:]`；Dataset 只把 PAD 位置设为 `-100`，`labels = input_ids.clone()` 并没有右移。
- `stage4_generate.py` 的 `[4/4]` 输出**必然是乱码**，那是脚本故意打印的「随机初始化对照组」。看到乱码不要判定训练失败，判断依据是 `[3/4]` 与 `loss_log.tsv`。
- SFT 默认用官方 `pretrain_768.pth`，是因为网页第 2 章起点固定 768/8 层，而降配 stage3 产出 384/6 层，结构不兼容。
- `seq_len=512` 时真实语料约 **59% 算力花在 padding** 上（抽样 p50≈211 / p90≈378 / mean≈237），这是缩短到 256 的依据。
- 64M 级模型会有事实错误：官方 SFT 权重实测把「1+1」答成 3。SFT 赋予的是对话行为，不是知识。
- 预训练模型只会续写、不会对话，这是网页的设计，不是 bug。

---

## 6. 硬约束 / 不要做

- ❌ 不代跑正式训练；不覆盖用户真实的 `loss_log.tsv`、`checkpoints/`、`sft_checkpoints/`、`data/`。
- ❌ 不提交 `.venv/`、`data/`、`*.pth`、语料、分词器、loss 日志（`.gitignore` 已排除，改动时别误删规则）。
- ❌ 不改许可证为 MIT/Apache；不删除 `LICENSE` / `NOTICE.md` / `examples/`；不用于商业场景。
- ❌ 不把设备检测退回单后端；不重新引入 `curl` 子进程、`/dev/null`、`.half()` 保存、`num_workers>0`。这四项都是跨平台改造时特意移除的。
- ❌ 不在公开仓库里写本机绝对路径、公司邮箱、代理地址、API key。提交前做残留扫描。
- ❌ 不把 ModelScope 下载源换成需要鉴权的地址；不改 `shared/download.py` 的 `GET + Range: bytes=0-0` 探测方式 —— 该 CDN 的 HEAD 在 302 后不返回 `Content-Range`。
- ❌ 不用全局 git 身份提交公开仓库。

---

## 7. 改代码时的同步清单

| 改动 | 必须同步 |
|---|---|
| `config.py` 的超参 / 档位 / 设备逻辑 | README 的平台与实测表、GUIDE 的档位与差异表、COMPARISON 的超参表、NOTICE 的修改摘要 |
| 新增 / 删除文件 | NOTICE 引用清单、README 目录结构、`.gitignore`（若为产物） |
| 下载逻辑 | `shared/download.py` + `stage0_setup.py` + `stage5_sft_data.py` |
| 训练循环 | `stage3_train.py` 与 `stage6_sft_train.py` 通常成对修改 |
| 任何共有代码 | 发布仓库 ↔ 本地运行目录，逐字节一致 |
| 实测数字变化 | COMPARISON 第八节、README 第八节、GUIDE 的实测段落，并注明测试平台 |

提交前固定跑一遍第 8 节的全部检查。

---

## 8. 验证方法（可直接复制）

```bash
# [1] 语法 + 转义警告（含 docstring 里的 Windows 路径）
python3 -W error::SyntaxWarning -c "
import ast,pathlib,warnings
fs=[f for f in pathlib.Path('.').rglob('*.py') if '.git' not in f.parts]
for f in fs:
    with warnings.catch_warnings():
        warnings.simplefilter('error'); ast.parse(f.read_text(encoding='utf-8'), filename=str(f))
print(len(fs), 'files OK')"

# [2] 本机痕迹 / 敏感信息（按需补充你所在组织的邮箱域名、内网地址等）
#     注：AGENTS.md 自身包含这些模式字面串，会命中自己，属预期，逐行确认即可
grep -RInE '/Users/|/tmp/|sk-[A-Za-z0-9]' . --exclude-dir=.git

# [3] 设备选择逻辑（mock 各种硬件组合，不需要真 GPU）
#     config.pick_device(forced) 可接受 '' / cuda / mps / cpu；
#     patch config._device_flags 返回 (cuda_ok, mps_ok) 后断言 .type。

# [4] 隔离短程实跑：复制到 /tmp 新目录，只放少量样本 + 两个分词器文件
cd <临时目录>
LAB_DEVICE=cpu  PRESET=quick LAB_LIMIT_STEPS=9 <venv-python> stage3_train.py   # 9 步会触发 optimizer.step
PRESET=quick LAB_LIMIT_STEPS=2 <venv-python> stage3_train.py                  # MPS 无回归
LAB_DEVICE=cpu SFT_BASE=official LAB_SFT_LIMIT_STEPS=2 <venv-python> stage6_sft_train.py

# [5] 推理工具只读性：确认不写 checkpoint / 不覆盖 loss 日志
<venv-python> play_sft.py --list
<venv-python> play_sft.py --greedy --len 24 "1+1等于几？"
printf ':params\n:q\n' | <venv-python> play_sft.py

# [6] 两份目录共有文件一致性：对第 2 节列出的文件逐个 diff -q
```

**不要**在项目根目录直接跑 `LAB_LIMIT_STEPS` 冒烟：`stage3` 开头会 `LOSS_LOG.write_text(...)` **清空用户真实 loss 曲线**，`stage6` 同理，且会写出新 checkpoint。

---

## 9. 常见坑（都是实测踩过的）

| 现象 | 原因 / 处理 |
|---|---|
| `Path.mkdir(exist_ok=True)` 报 EEXIST | 某些 macOS 沙箱的 `sitecustomize` 会拦截；`config.py` 已改为 `if not exists: mkdir()`，别改回去 |
| MPS 上 FP16 权重算子报错 / 静默出错 | 全程用 FP32 保存与加载，不调用 `.half()` |
| DataLoader 多进程 pickle 失败 | `NUM_WORKERS=0` 固定，macOS spawn 与 Windows 多进程都靠它规避 |
| Windows 找不到 `.venv/bin/python` | 用 `setup.ps1`，解释器是 `.venv\Scripts\python.exe` |
| zsh 里 `for f in $FILES` 只循环一次 | zsh 默认不做单词分割；用字面列表或 `${=FILES}` |
| 下载探测拿不到 `Content-Range` | 必须用 GET + `Range: bytes=0-0`，HEAD 在 302 后不返回 |
| 分段下载首/尾行 `UnicodeDecodeError` | 片段边界是半行，`stage0` 已按 JSON 可解析性过滤 |
| Intel Mac 装不上 torch | `torch 2.15.0` 只有 `macosx_14_0_arm64` wheel，需自行换版本 |
| 境外下载慢 | 语料走 ModelScope 国内镜像，属已知限制 |
| 训练明显变慢 | 无风扇 Mac 热降频、接电状态、swap 压力；第 200 步脚本会打印实测外推 |

---

## 10. 未完成 / 未验证（接手时优先看这里）

- [ ] NVIDIA GPU 真机上的完整训练与耗时（CUDA 分支目前只有逻辑测试）
- [ ] Windows 原生端到端跑通（脚本与安装脚本已适配，未实机验证）
- [x] ~~SFT 完整 1,875 步训练与最终对话质量~~（2026-11-10 实跑完成，见第 5 节）
- [ ] Intel Mac / macOS x86_64 的支持方案
- [ ] ModelScope 境外访问的可选镜像

**这些留给用户自己决定是否推进**；在动手前先确认，不要默认「顺手跑一下」。

---

## 11. 协作约定

- 用**中文**回复；结论先行，配合表格、检查清单与验证步骤。
- 下结论前先**实际验证**（读日志、跑命令、查元数据），不要只凭推理。
- 动手前先排查当前状态；修改前确认影响范围；改完明确列出改动的文件与验证结果。
- 用户偏好自己按网页一步步执行实验流程，助手负责整理材料、校准与排障，**不代跑训练**。
- 如同时改了两份目录，交付前用 `diff -q` 逐个确认一致。
