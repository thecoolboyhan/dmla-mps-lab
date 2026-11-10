# Windows PowerShell 安装脚本，与 setup.sh 等价。
# 若执行策略阻止脚本，用下面这种方式运行：
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "未找到 uv。请先在 PowerShell 里执行：" -ForegroundColor Yellow
    Write-Host '  irm https://astral.sh/uv/install.ps1 | iex' -ForegroundColor Yellow
    Write-Host "装好后重新打开终端，再运行本脚本。" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/3] 准备 Python 3.14 ..."
uv python install 3.14

Write-Host "[2/3] 创建 .venv ..."
uv venv --python 3.14 .venv

Write-Host "[3/3] 安装依赖（--no-cache 保证可重复、不残留缓存）..."
$env:PYTHONDONTWRITEBYTECODE = "1"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
uv pip install --no-cache --python $venvPython -r requirements.txt

Write-Host ""
Write-Host "完成。之后所有脚本都用虚拟环境里的解释器运行，例如："
Write-Host '  .venv\Scripts\python.exe stage0_setup.py'
Write-Host '  .venv\Scripts\python.exe stage1_tokenizer.py'
Write-Host ""
Write-Host '切换档位：  $env:PRESET="quick"; .venv\Scripts\python.exe stage3_train.py'
Write-Host '指定后端：  $env:LAB_DEVICE="cpu";  .venv\Scripts\python.exe stage3_train.py'
