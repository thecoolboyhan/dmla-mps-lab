"""
对应网页「工程实训：SFT 模型对话实验」的实验准备 + SFT 数据集两节。

网页这一步由 `dmla data` 下载 MiniMind SFT 语料（5 万条 / 90MB），
并以上一章产物 `pretrain_768.pth` 作为 SFT 起点。本机做三件事：

  1. 直链下载 SFT 语料与官方预训练权重（等价于网页第 1 章应该产出的文件）
  2. 按 token 长度筛出「一问一答能完整放进 seq」的样本，再随机抽子集
  3. 展示 ChatML 格式与 labels 掩码 —— 本章最核心的教学点

[本机适配] 只改下载方式（HTTP 直链代替 dmla data / Git LFS）与序列长度，
SFTDataset 的算法在 shared/sft_dataset.py 里与网页逐行一致。
"""
import json
import random
import subprocess
import sys

from config import (CHATML_TEMPLATE, SFT_BASE, SFT_CORPUS, SFT_CORPUS_URL,
                    SFT_MAX_SEQ_LEN, SFT_OFFICIAL_PRETRAIN, SFT_OFFICIAL_PRETRAIN_URL,
                    SFT_SAMPLES, SFT_SUBSET, TOKENIZER_DIR, sft_banner)
from transformers import AutoTokenizer

random.seed(42)


def download(url, out, label):
    if out.exists() and out.stat().st_size > 1024 * 1024:
        print(f"  [已有] {label:28s} {out.stat().st_size/1024/1024:7.1f} MB")
        return
    print(f"  [下载] {label} ...", flush=True)
    r = subprocess.run(["curl", "-sSL", "--max-time", "300", "-o", str(out),
                        "-w", "%{http_code}", url], capture_output=True, text=True)
    code = r.stdout.strip()
    if code not in ("200", "206") or not out.exists() or out.stat().st_size < 1024:
        if out.exists():
            out.unlink()
        raise SystemExit(f"下载失败: HTTP {code} {r.stderr.strip()}")
    print(f"  [完成] {label:28s} {out.stat().st_size/1024/1024:7.1f} MB")


def build_subset():
    """网页用全量 5 万条 + seq=768；本机 seq=256，必须丢掉会被截断的长对话，
    否则 assistant 回答被切一半，labels 里学不到完整的「回答该怎么收尾」。"""
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    lines = SFT_CORPUS.read_text(encoding="utf-8", errors="replace").splitlines()
    good, too_long, bad = [], 0, 0
    for line in lines:
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        conv = d.get("conversations")
        if not conv:
            bad += 1
            continue
        text = "".join(
            f"<|im_start|>{m.get('role','')}\n{m.get('content','')}<|im_end|>\n"
            for m in conv
        )
        # 留 48 token 余量给 pre_processing_chat 可能补的 system prompt
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        if n <= SFT_MAX_SEQ_LEN - 48:
            good.append(line.strip())
        else:
            too_long += 1
    print(f"  语料 {len(lines):,} 行 | 可完整放进 seq={SFT_MAX_SEQ_LEN} 的 {len(good):,} 条"
          f" | 过长丢弃 {too_long:,} | 无效 {bad:,}")
    if len(good) < 200:
        raise SystemExit("可用短对话太少，请把 SFT_MAX_SEQ_LEN 调大")
    chosen = random.sample(good, min(SFT_SAMPLES, len(good)))
    SFT_SUBSET.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(f"  已写出 {SFT_SUBSET.name}: {len(chosen):,} 条 / "
          f"{SFT_SUBSET.stat().st_size/1024/1024:.1f} MB")
    return chosen


def main():
    sft_banner()
    print("[1/4] 环境自检")
    import torch
    from config import DEVICE
    print(f"  Python {sys.version.split()[0]} | torch {torch.__version__} | 设备 {DEVICE}")

    print("\n[2/4] 下载 SFT 语料与起点权重（对应网页 `dmla data` 与第 1 章产物）")
    download(SFT_CORPUS_URL, SFT_CORPUS, "sft_t2t_tiny.jsonl")
    if SFT_BASE == "official":
        download(SFT_OFFICIAL_PRETRAIN_URL, SFT_OFFICIAL_PRETRAIN, "pretrain_768.pth")
        print("  ↑ 这就是网页第 1 章训练完成后应该生成的那个文件；")
        print("    本机降配版第 1 章产出的是 384/6 层，结构接不上 SFT 章，故在此补齐。")
    else:
        from config import FINAL_CKPT
        print(f"  [local] 起点将使用你自己 stage3 的产物: {FINAL_CKPT.name}")
        if not FINAL_CKPT.exists():
            print("  ⚠ 该文件还不存在，请先跑完 stage3_train.py，或改用默认 SFT_BASE=official")

    print("\n[3/4] 构建训练子集（过滤会被 seq 截断的长对话）")
    build_subset()

    print("\n[4/4] ChatML 格式与 labels 掩码（本章最核心的差别）")
    from shared.sft_dataset import SFTDataset
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    tok.chat_template = CHATML_TEMPLATE
    ds = SFTDataset(str(SFT_SUBSET), tok, max_length=SFT_MAX_SEQ_LEN)
    ids, labels = ds[0]
    prompt = ds.create_chat_prompt(ds.samples[0]["conversations"])
    print(f"  ChatML 原文前 240 字:\n    {prompt[:240].replace(chr(10), ' ⏎ ')}")
    print(f"\n  input_ids 形状 {tuple(ids.shape)}；labels 中 -100 有 "
          f"{int((labels == -100).sum()):,} 个 / 共 {len(labels)} 个")
    print(f"  → 只有 {int((labels != -100).sum()):,} 个位置参与 loss "
          f"({(labels != -100).float().mean()*100:.1f}%)，全是 assistant 的回答。")
    print("  → 预训练 Dataset 是「每个位置都监督」；SFT 把用户提问/系统提示全屏蔽，")
    print("    模型因此学会「用户说什么不用预测，只需在 assistant 位置给出回答」。")
    print("  → 这一步就是网页所说「从续写到对话的转变」的机制来源。")

    print("\n准备完成。下一步： .venv/bin/python stage6_sft_train.py")
    print("（建议同时打开网页『SFT 模型对话实验』对照阅读）")


if __name__ == "__main__":
    main()
