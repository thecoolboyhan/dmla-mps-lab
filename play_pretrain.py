"""
交互续写：用你自己 stage3 训出来的权重，输入一句话，看模型往下接什么。

  .venv/bin/python play_pretrain.py                 # 进入交互
  .venv/bin/python play_pretrain.py "人工智能是"     # 只跑一次就退出

这是**预训练 base 模型**：只会顺着你的话续写，不会回答问题，也不记得上一句。
想让它会对话，继续跑 stage5 → stage6 → stage7（网页 SFT 章）。

交互命令：
  :len 128      改每次生成的 token 数
  :temp 0.7     改温度（越低越保守，越高越乱）
  :top 0.9      改 top_p
  :params       查看当前采样参数
  :help         帮助
  :q / exit     退出
"""
import sys

import torch
from transformers import AutoTokenizer

from config import (DEVICE, FINAL_CKPT, HIDDEN_SIZE, NUM_LAYERS, PRESET,
                    SAVE_DIR, TOKENIZER_DIR)
from shared.mini_mind_config import MiniMindConfig, MiniMindForCausalLM

MAX_NEW = 64
TEMPERATURE = 0.85
TOP_P = 0.85
TOP_K = 50
REPETITION_PENALTY = 1.2


def load_model():
    ckpt = FINAL_CKPT
    if not ckpt.exists():
        cands = sorted(SAVE_DIR.glob(f"pretrain_{PRESET}_*.pth"))
        if not cands:
            raise SystemExit("没有找到任何 checkpoint，请先跑完 stage3_train.py")
        ckpt = cands[-1]
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=HIDDEN_SIZE,
                                               num_hidden_layers=NUM_LAYERS))
    w = torch.load(ckpt, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(w, strict=False)
    except RuntimeError as e:
        raise SystemExit(f"权重结构与当前 PRESET ({PRESET}) 不匹配，请确认训练时用的是同一档。\n{e}")
    model = model.to(DEVICE).eval()
    return tok, model, ckpt


def once(tok, model, text, max_new, temp, top_p):
    inputs = tok(tok.bos_token + text, return_tensors="pt", truncation=True,
                 max_length=512).to(DEVICE)
    with torch.no_grad():
        try:
            out = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new,
                temperature=temp, top_p=top_p, top_k=TOP_K, do_sample=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                repetition_penalty=REPETITION_PENALTY,
            )
        except (RuntimeError, TypeError):
            out = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new, do_sample=True,
                temperature=temp, top_p=top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def banner(ckpt, model):
    print("=" * 68)
    print("  交互续写 · 你自己训出来的预训练模型")
    print(f"  权重 {ckpt.name}  |  {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
          f"  |  {DEVICE}  |  档位 {PRESET}")
    print("  ⚠ 这是 base 模型：只会顺着你的话往下续写，不会回答问题、不记得上一句。")
    print("     想让它「会说话」，请继续 stage5 → stage6 → stage7（SFT）。")
    print("=" * 68)
    print(f"  采样参数：max_new_tokens={MAX_NEW}  temperature={TEMPERATURE}  top_p={TOP_P}")
    print("  命令：:len 128 / :temp 0.7 / :top 0.9 / :params / :help / :q 退出")
    print()


def main():
    tok, model, ckpt = load_model()
    max_new, temp, top_p = MAX_NEW, TEMPERATURE, TOP_P

    # 单次模式：命令行直接带一句话
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"输入: {text}")
        print(f"续写: {once(tok, model, text, max_new, temp, top_p)}")
        return

    banner(ckpt, model)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            continue
        if line.lower() in (":q", "exit", "quit"):
            print("bye")
            break
        if line == ":help":
            print("  :len 128    改生成长度\n"
                  "  :temp 0.7   改温度\n"
                  "  :top 0.9    改 top_p\n"
                  "  :params     查看当前参数\n"
                  "  :q / exit   退出\n"
                  "  直接输入任意文字 → 模型顺着往下续写")
            continue
        if line == ":params":
            print(f"  max_new_tokens={max_new}  temperature={temp}  top_p={top_p}  "
                  f"top_k={TOP_K}  repetition_penalty={REPETITION_PENALTY}")
            continue
        if line.startswith(":len "):
            try:
                max_new = max(1, min(512, int(line.split()[1])))
                print(f"  max_new_tokens = {max_new}")
            except (ValueError, IndexError):
                print("  用法：:len 128")
            continue
        if line.startswith(":temp "):
            try:
                temp = float(line.split()[1])
                print(f"  temperature = {temp}")
            except (ValueError, IndexError):
                print("  用法：:temp 0.7")
            continue
        if line.startswith(":top "):
            try:
                top_p = float(line.split()[1])
                print(f"  top_p = {top_p}")
            except (ValueError, IndexError):
                print("  用法：:top 0.9")
            continue
        if line.startswith(":"):
            print("  未知命令，输入 :help 查看")
            continue

        try:
            out = once(tok, model, line, max_new, temp, top_p)
        except KeyboardInterrupt:
            print("\n  (已中断本次生成)")
            continue
        print(f"  {out}\n")


if __name__ == "__main__":
    main()
