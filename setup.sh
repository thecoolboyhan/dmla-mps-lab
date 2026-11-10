#!/usr/bin/env bash
# 在本目录建一套隔离的 Python 环境。不改动系统 Python，也不改动你现有的任何环境。
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "未找到 uv。请先安装：curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "[1/3] 准备 Python 3.14 ..."
uv python install 3.14

echo "[2/3] 创建 .venv ..."
uv venv --python 3.14 .venv

echo "[3/3] 安装依赖（--no-cache 保证可重复、不残留缓存）..."
PYTHONDONTWRITEBYTECODE=1 TMPDIR="${TMPDIR:-/tmp}" \
  uv pip install --no-cache --python .venv/bin/python -r requirements.txt

echo
echo "完成。之后所有脚本都用 .venv/bin/python 运行，例如："
echo "  .venv/bin/python stage0_setup.py"
echo "  .venv/bin/python stage1_tokenizer.py"
echo 'Windows 请改用 setup.ps1，解释器路径为 .venv\Scripts\python.exe。'
