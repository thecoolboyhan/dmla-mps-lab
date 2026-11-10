"""
对应网页「实验准备」一节：验证语料与分词器文件、准备数据。

网页里这一步由 DMLA 沙箱的 `dmla data` 完成（全量 1.2 GB + Git LFS）。
本项目改用 Python 标准库的 HTTP Range 从 ModelScope 分段采样，只下 DOWNLOAD_MB 这么多，
再随机抽 SUBSET_SAMPLES 行写成 train_subset.jsonl —— 之后的 Dataset 代码一行不改。
下载器在 shared/download.py，不依赖 curl / Git LFS / Docker，三个平台一致。

额外做一件网页没做但对理解很有帮助的事：统计真实语料的 token 长度分布，
让你亲眼看到「固定 padding 到 max_seq_len 到底浪费了多少算力」。
"""
import json
import random
import sys
from pathlib import Path

from config import (BASE_URL, CORPUS_SAMPLE, CORPUS_URL, DATA_DIR, DEVICE,
                    DEVICE_TYPE, DOWNLOAD_MB, MAX_SEQ_LEN, PY_CMD,
                    SUBSET_SAMPLES, TOKENIZER_DIR, TRAIN_SUBSET, banner)
from shared.download import fetch as http_fetch, probe_size

random.seed(42)


def ensure_tokenizer():
    for name in ("tokenizer.json", "tokenizer_config.json"):
        dst = TOKENIZER_DIR / name
        if dst.exists() and dst.stat().st_size > 1000:
            print(f"  [已有] {name}  {dst.stat().st_size:,} bytes")
            continue
        code = http_fetch(f"{BASE_URL}/{name}", dst)
        if code != 200 or not dst.exists():
            raise SystemExit(f"下载 {name} 失败: HTTP {code}；请检查网络或代理设置")
        print(f"  [下载] {name}  {dst.stat().st_size:,} bytes")


def build_corpus_pool() -> bytes:
    if CORPUS_SAMPLE.exists() and CORPUS_SAMPLE.stat().st_size > DOWNLOAD_MB * 1024 * 1024 * 0.8:
        print(f"  [已有] 语料采样池 {CORPUS_SAMPLE.stat().st_size/1024/1024:.0f} MB")
        return CORPUS_SAMPLE.read_bytes()

    total = probe_size(CORPUS_URL)
    if total:
        print(f"  远端语料总大小: {total/1024/1024/1024:.2f} GB")
    seg = max(4, DOWNLOAD_MB // 20)          # 每段约 20MB
    per = DOWNLOAD_MB * 1024 * 1024 // seg
    tmp = DATA_DIR / ".segments.tmp"
    blobs = []
    print(f"  分 {seg} 段采样，每段 {per/1024/1024:.0f} MB（跨文件不同位置，避免只取到开头）...")
    for i in range(seg):
        start = int(total * i / seg) if total else i * per * 3
        end = start + per - 1
        try:
            code = http_fetch(CORPUS_URL, tmp, f"{start}-{end}")
        except Exception as e:
            print(f"    segment {i+1}/{seg}: 跳过 ({type(e).__name__}: {e})")
            continue
        if code == 206 and tmp.exists() and tmp.stat().st_size > 1000:
            blobs.append(tmp.read_bytes())
            print(f"    segment {i+1}/{seg}: {tmp.stat().st_size/1024/1024:.0f} MB  ok")
        else:
            print(f"    segment {i+1}/{seg}: 跳过 (HTTP {code})")
    tmp.unlink(missing_ok=True)

    if not blobs:  # 降级：直接取文件开头一段
        print("  分段采样不可用，降级为单段下载 ...")
        code = http_fetch(CORPUS_URL, tmp, f"0-{DOWNLOAD_MB*1024*1024-1}")
        if code not in (200, 206):
            raise SystemExit(f"语料下载失败: HTTP {code}；请检查网络或代理设置")
        blobs = [tmp.read_bytes()]
        tmp.unlink(missing_ok=True)

    raw = b"\n".join(blobs)
    # 每个片段的首尾行通常不完整，统一在下面按 JSON 可解析性过滤
    CORPUS_SAMPLE.write_bytes(raw)
    print(f"  语料采样池已保存: {CORPUS_SAMPLE.stat().st_size/1024/1024:.0f} MB")
    return raw


def main():
    banner()
    print("\n[1/4] 环境自检")
    import torch, transformers
    _mps = getattr(torch.backends, "mps", None)
    print(f"  Python       {sys.version.split()[0]}")
    print(f"  torch        {torch.__version__}")
    print(f"  transformers {transformers.__version__}")
    print(f"  CUDA 可用    {torch.cuda.is_available()}  |  MPS 可用 {bool(_mps and _mps.is_available())}")
    print(f"  选定后端     {DEVICE_TYPE.upper()}  →  {DEVICE}")
    if DEVICE_TYPE == "cpu":
        print("  ⚠ 无可用 GPU，训练走 CPU，耗时长一个数量级；建议先 PRESET=quick 验证链路")

    print("\n[2/4] 下载分词器文件")
    ensure_tokenizer()

    print("\n[3/4] 构建训练子集")
    raw = build_corpus_pool()
    lines = []
    for raw_line in raw.split(b"\n"):
        try:
            s = raw_line.decode("utf-8").strip()
            if not s:
                continue
            d = json.loads(s)
            if isinstance(d, dict) and d.get("text", "").strip():
                lines.append(json.dumps(d, ensure_ascii=False))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue          # 片段边界的半行，直接丢弃
    print(f"  采样池中有效样本: {len(lines):,} 条")
    if len(lines) < SUBSET_SAMPLES:
        print(f"  ⚠ 少于目标 {SUBSET_SAMPLES:,} 条，将使用全部样本")
        chosen = lines
    else:
        chosen = random.sample(lines, SUBSET_SAMPLES)
    TRAIN_SUBSET.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(f"  已写出 {TRAIN_SUBSET.name}: {len(chosen):,} 条 / "
          f"{TRAIN_SUBSET.stat().st_size/1024/1024:.1f} MB")

    print("\n[4/4] 语料 token 长度分布（教学观察点）")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    sample = random.sample(chosen, min(2000, len(chosen)))
    lens = []
    for s in sample:
        text = json.loads(s)["text"]
        lens.append(len(tok(text, add_special_tokens=False, truncation=True,
                            max_length=4096)["input_ids"]))
    lens.sort()
    p = lambda q: lens[min(len(lens) - 1, int(len(lens) * q))]
    print(f"  p50={p(.5)}  p90={p(.9)}  p95={p(.95)}  mean={sum(lens)/len(lens):.0f}  max={max(lens)}")
    print(f"  当前档位 max_seq_len={MAX_SEQ_LEN}")
    kept = sum(min(x, MAX_SEQ_LEN - 2) for x in lens)
    print(f"  有效 token 保留率 {kept/sum(lens)*100:.1f}%  |  "
          f"padding 计算浪费 {100-kept/(len(lens)*(MAX_SEQ_LEN-2))*100:.1f}%")
    print("\n  → 网页原版 seq=512 时这一项约 59%，这就是缩短序列几乎不损效果的依据。")
    print(f"\n准备完成。下一步： {PY_CMD} stage1_tokenizer.py")
    print("（建议同时打开网页『第一阶段：分词器』对照阅读）")


if __name__ == "__main__":
    main()
