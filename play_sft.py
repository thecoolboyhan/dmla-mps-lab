r"""
交互对话：用你自己 stage6 训出来的 SFT 权重，输入问题，看它怎么答。

  <venv-python> play_sft.py                              # 进入交互
  <venv-python> play_sft.py "1+1等于几？"                 # 单次提问
  <venv-python> play_sft.py --list                       # 列出所有可对比的权重
  <venv-python> play_sft.py --compare "1+1等于几？"       # 起点 / 自己训的 / 官方 三方对照
  <venv-python> play_sft.py --ckpt official "你好"        # 换官方 SFT 权重回答
  <venv-python> play_sft.py --ckpt base "你好"            # 换 SFT 起点权重（没学过对话格式）
  <venv-python> play_sft.py --ckpt sft_official_step1000.pth "你好"   # 对比中间 checkpoint
  <venv-python> play_sft.py --batch questions.txt --out /tmp/result.tsv # 批量跑自定义问题集
  可选参数：--len 96 --temp 0.85 --top 0.85 --sys "你是一个有帮助的AI助手。" --greedy

<venv-python> 在 macOS/Linux 是 .venv/bin/python，
在 Windows PowerShell 是 .venv\Scripts\python.exe。

只读权重、不训练、不写 checkpoint。采样参数默认与 stage7_sft_chat.py 一致。
⚠ 这是 64M 级小模型：SFT 只教它「按 user/assistant 格式说话」，不增加知识与推理能力，
  回答会一本正经地胡说（例如 1+1=3）属于正常现象，是本章的教学结论之一。

交互命令：
  :len 128      改每次生成的 token 数        :temp 0.7    改温度
  :top 0.9      改 top_p                     :greedy      切换贪心 / 采样
  :sys 文本     改 system 提示                :multi       开关多轮上下文（默认关）
  :reset        清空多轮历史                  :ckpt 别名   热切换权重（见 :list）
  :list         列出可用权重                  :params      查看当前配置
  :help         帮助                         :q / exit    退出
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from config import (CHATML_TEMPLATE, DEVICE, SFT_BASE, SFT_BASE_CKPT,
                    SFT_FINAL_CKPT, SFT_HIDDEN, SFT_LAYERS, SFT_OFFICIAL_SFT,
                    SFT_SAVE_DIR, TOKENIZER_DIR)
from shared.mini_mind_config import MiniMindConfig, MiniMindForCausalLM

MAX_NEW = 96
TEMPERATURE = 0.85
TOP_P = 0.85
TOP_K = 50
REPETITION_PENALTY = 1.2
DEFAULT_SYS = "你是一个有帮助的AI助手。"

# 权重别名：mine = 你自己训出来的；base = SFT 起点；official = 网页官方标准答案
ALIASES = {
    "mine": SFT_FINAL_CKPT,
    "self": SFT_FINAL_CKPT,
    "local": SFT_FINAL_CKPT,
    "base": SFT_BASE_CKPT,
    "official": SFT_OFFICIAL_SFT,
}


def resolve_ckpt(name: str) -> Path:
    """支持别名、sft_checkpoints 里的文件名、以及任意存在的路径。"""
    key = name.strip().lower()
    if key in ALIASES:
        p = ALIASES[key]
    else:
        p = Path(name)
        if not p.is_absolute() and not p.exists():
            p = SFT_SAVE_DIR / name
    if not p.exists():
        available = ", ".join(sorted(ALIASES))
        raise ValueError(f"找不到权重 {name!r}。可用别名：{available}；"
                         f"或 {SFT_SAVE_DIR} 下的文件名 / 绝对路径（用 --list 查看）")
    return p


def list_ckpts() -> None:
    print("=" * 68)
    print("  可对比的权重（结构必须与当前 SFT_BASE 一致）")
    print(f"  当前 SFT_BASE={SFT_BASE}  →  模型 {SFT_HIDDEN}/{SFT_LAYERS} 层")
    print("=" * 68)
    for alias, p in ALIASES.items():
        size = f"{p.stat().st_size / 1024 / 1024:.1f} MB" if p.exists() else "（不存在）"
        note = ""
        if alias in ("mine", "self", "local"):
            note = "  ← 你自己 stage6 训出来的"
        elif alias == "base":
            note = "  ← SFT 起点，只会续写"
        elif alias == "official":
            note = "  ← 网页官方标准答案"
        print(f"  {alias:<10} {p.name:<38} {size:>10}{note}")
    print("-" * 68)
    print("  sft_checkpoints/ 下的其它 checkpoint（可直接把文件名传给 --ckpt / :ckpt）：")
    cands = sorted(SFT_SAVE_DIR.glob("*.pth"))
    if not cands:
        print("    （空）")
    for p in cands:
        print(f"    {p.name}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    print()


def load_model(path: Path):
    cfg = MiniMindConfig(hidden_size=SFT_HIDDEN, num_hidden_layers=SFT_LAYERS)
    model = MiniMindForCausalLM(cfg)
    weights = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise ValueError(f"权重结构不匹配 (missing={len(missing)}, unexpected={len(unexpected)})。"
                         f"请确认训练时与现在用的是同一个 SFT_BASE / PRESET")
    del weights
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    return model.to(DEVICE).eval(), n_params


def answer(tok, model, question: str, history, sys_prompt: str,
           max_new: int, temp: float, top_p: float, greedy: bool) -> str:
    messages = [{"role": "system", "content": sys_prompt}]
    messages += list(history)
    messages += [{"role": "user", "content": question}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    kw = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        repetition_penalty=REPETITION_PENALTY,
    )
    if greedy:
        kw["do_sample"] = False
    else:
        kw.update(do_sample=True, temperature=temp, top_p=top_p, top_k=TOP_K)
    with torch.no_grad():
        try:
            out = model.generate(**kw)
        except (RuntimeError, TypeError):
            # 兼容不同 transformers 版本对采样参数的支持差异
            kw.pop("top_k", None)
            if greedy:
                kw.pop("temperature", None)
                kw.pop("top_p", None)
            out = model.generate(**kw)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def make_tok():
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    tok.chat_template = CHATML_TEMPLATE
    return tok


def clean(text: str) -> str:
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def run_compare(tok, question: str, args) -> None:
    print("=" * 68)
    print(f"  同一问题三方对照：{question}")
    print("  全部用 ChatML 提问 → 只有学过 assistant 结构的权重才会「回答」")
    print("=" * 68)
    for alias in ("base", "mine", "official"):
        try:
            p = resolve_ckpt(alias)
        except ValueError as e:
            print(f"[{alias}] 跳过：{e}")
            continue
        model, _ = load_model(p)
        print(f"\n[{alias}]  {p.name}")
        print(f"  {clean(answer(tok, model, question, [], args.sys, args.len, args.temp, args.top, args.greedy))}")
        del model
    print("\n  → base 只会顺着模板往下续写；mine / official 会给出助手式回答。")
    print("    回答内容对不对是另一回事：SFT 不增加知识，小模型仍会事实错误。")


def run_batch(tok, args) -> None:
    if not args.batch.exists():
        raise SystemExit(f"问题集不存在：{args.batch}")
    questions = [l.strip() for l in args.batch.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
    if not questions:
        raise SystemExit(f"问题集里没有有效问题：{args.batch}（空行与 # 开头的行会被跳过）")
    ckpt = resolve_ckpt(args.ckpt)
    model, _ = load_model(ckpt)
    rows = ["question\tanswer\tckpt\ttemperature\ttop_p\tmax_new\tgreedy"]
    print(f"批量测试 {len(questions)} 条 | 权重 {ckpt.name} | 设备 {DEVICE}", file=sys.stderr)
    for i, q in enumerate(questions, 1):
        a = clean(answer(tok, model, q, [], args.sys, args.len, args.temp, args.top, args.greedy))
        rows.append(f"{clean(q)}\t{a}\t{ckpt.name}\t{'-' if args.greedy else args.temp}"
                    f"\t{'-' if args.greedy else args.top}\t{args.len}\t{args.greedy}")
        print(f"  [{i}/{len(questions)}] {q[:40]}", file=sys.stderr)
    out = "\n".join(rows) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out, encoding="utf-8")
        print(f"已写入 {args.out}（{len(questions)} 条）")
    else:
        sys.stdout.write(out)


def banner(ckpt: Path, n_params: float) -> None:
    print("=" * 68)
    print("  交互对话 · 你自己 stage6 训出来的 SFT 模型")
    print(f"  权重 {ckpt.name}  |  {n_params:.2f}M  |  {DEVICE}  |  SFT_BASE={SFT_BASE}")
    print("  ⚠ 64M 小模型：会按格式回答，但内容可能一本正经地胡说，属正常现象。")
    print("=" * 68)
    print("  命令：:len / :temp / :top / :greedy / :sys / :multi / :reset")
    print("        :ckpt 别名（热切换权重） / :list / :params / :help / :q 退出")
    print()


def repl(tok, model, ckpt: Path, n_params: float, args) -> None:
    max_new, temp, top_p = args.len, args.temp, args.top
    greedy, sys_prompt, multi = args.greedy, args.sys, False
    history: list[dict] = []
    banner(ckpt, n_params)
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            continue
        low = line.lower()
        if low in (":q", "exit", "quit"):
            print("bye")
            break
        if low == ":help":
            print(__doc__.strip().split("交互命令：")[-1].strip())
            continue
        if low == ":params":
            print(f"  权重 {ckpt.name}  |  {n_params:.2f}M  |  {DEVICE}")
            print(f"  max_new_tokens={max_new}  temperature={'-' if greedy else temp}  "
                  f"top_p={'-' if greedy else top_p}  top_k={TOP_K}  greedy={greedy}")
            print(f"  repetition_penalty={REPETITION_PENALTY}  multi={multi}  history={len(history)//2} 轮")
            print(f"  system = {sys_prompt}")
            continue
        if low == ":list":
            list_ckpts()
            continue
        if low == ":reset":
            history.clear()
            print("  多轮历史已清空")
            continue
        if low == ":greedy":
            greedy = not greedy
            print(f"  greedy = {greedy}（{'贪心，结果可复现' if greedy else '采样'}）")
            continue
        if low == ":multi":
            multi = not multi
            if not multi:
                history.clear()
            print(f"  多轮上下文 = {multi}" + ("（历史已清空）" if not multi else ""))
            continue
        if low.startswith(":len "):
            try:
                max_new = max(1, min(512, int(line.split()[1])))
                print(f"  max_new_tokens = {max_new}")
            except (ValueError, IndexError):
                print("  用法：:len 128")
            continue
        if low.startswith(":temp "):
            try:
                temp = float(line.split()[1])
                print(f"  temperature = {temp}")
            except (ValueError, IndexError):
                print("  用法：:temp 0.7")
            continue
        if low.startswith(":top "):
            try:
                top_p = float(line.split()[1])
                print(f"  top_p = {top_p}")
            except (ValueError, IndexError):
                print("  用法：:top 0.9")
            continue
        if line.startswith(":sys "):
            sys_prompt = line[5:].strip() or DEFAULT_SYS
            print(f"  system = {sys_prompt}")
            continue
        if low.startswith(":ckpt "):
            name = line.split(None, 1)[1]
            try:
                p = resolve_ckpt(name)
                fresh, fresh_params = load_model(p)
            except ValueError as e:
                print(f"  {e}")
                continue
            del model
            model, ckpt, n_params = fresh, p, fresh_params
            history.clear()
            print(f"  已切换到 {p.name}（{n_params:.2f}M），多轮历史已清空")
            continue
        if line.startswith(":"):
            print("  未知命令，输入 :help 查看")
            continue
        try:
            out = answer(tok, model, line, history, sys_prompt, max_new, temp, top_p, greedy)
        except KeyboardInterrupt:
            print("\n  (已中断本次生成)")
            continue
        print(f"  {out}\n")
        if multi:
            history += [{"role": "user", "content": line},
                        {"role": "assistant", "content": out}]
            history = history[-8:]   # 只保留最近 4 轮，避免 512 上下文被挤爆


def parse_args():
    p = argparse.ArgumentParser(
        description="交互测试本地 SFT 模型（只读权重，不训练）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               f"  .venv/bin/python play_sft.py\n"
               f"  .venv/bin/python play_sft.py --compare \"1+1等于几？\"\n"
               f"  .venv/bin/python play_sft.py --batch questions.txt --out /tmp/result.tsv\n")
    p.add_argument("question", nargs="*", help="单次提问的问题；留空则进入交互模式")
    p.add_argument("--ckpt", default="mine",
                   help="权重别名 mine/base/official，或 sft_checkpoints 下的文件名 / 绝对路径")
    p.add_argument("--list", action="store_true", help="列出所有可对比的权重后退出")
    p.add_argument("--compare", action="store_true", help="同一问题跑 base/mine/official 三方对照")
    p.add_argument("--batch", type=Path, help="批量测试：每行一个问题的 txt 文件")
    p.add_argument("--out", type=Path, help="--batch 的结果输出 tsv；不写则打到 stdout")
    p.add_argument("--len", type=int, default=MAX_NEW, help="生成 token 数（默认 96）")
    p.add_argument("--temp", type=float, default=TEMPERATURE, help="温度（默认 0.85）")
    p.add_argument("--top", type=float, default=TOP_P, help="top_p（默认 0.85）")
    p.add_argument("--sys", default=DEFAULT_SYS, help="system 提示词")
    p.add_argument("--greedy", action="store_true", help="贪心解码，关闭随机采样，结果可复现")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        list_ckpts()
        return
    tok = make_tok()
    if args.batch:
        run_batch(tok, args)
        return
    if args.compare:
        run_compare(tok, " ".join(args.question) or "1+1等于几？", args)
        return
    try:
        ckpt = resolve_ckpt(args.ckpt)
        model, n_params = load_model(ckpt)
    except ValueError as e:
        raise SystemExit(str(e))
    if args.question:
        q = " ".join(args.question)
        print(f"权重: {ckpt.name}  |  设备: {DEVICE}")
        print(f"提问: {q}")
        print(f"回答: {clean(answer(tok, model, q, [], args.sys, args.len, args.temp, args.top, args.greedy))}")
        return
    try:
        repl(tok, model, ckpt, n_params, args)
    finally:
        del model


if __name__ == "__main__":
    main()
