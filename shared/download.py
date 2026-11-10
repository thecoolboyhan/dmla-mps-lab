"""
只依赖 Python 标准库的极简 HTTP 下载工具（本项目原创，非上游改编）。

网页原版用 `dmla data` + Git LFS 取数据；早期本机版改用外部 curl 子进程
并把探测输出写到 /dev/null —— 这两者在 Windows 上都不可用。这里换成 urllib：

  · 支持 HTTP Range 分段下载（ModelScope 直链返回 206 + Content-Range）
  · 流式写盘，不把上百 MB 的文件整块读进内存
  · 自动跟随 302 重定向；写完用 os.replace 原子替换，中断不留半文件
  · 无外部命令依赖，macOS / Linux / Windows 行为一致

注意：不用 HEAD 探测大小 —— 该 CDN 的 HEAD 在 302 之后不返回 Content-Range，
必须用 GET + Range: bytes=0-0（这是实测得出的结论）。
"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

UA = "dmla-mps-lab/1.0 (educational use; https://github.com/thecoolboyhan/dmla-mps-lab)"


def _open(url: str, byte_range: str | None, timeout: int):
    headers = {"User-Agent": UA}
    if byte_range:
        headers["Range"] = f"bytes={byte_range}"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def fetch(url: str, out: Path, byte_range: str | None = None,
          timeout: int = 300) -> int:
    """下载 url 到 out，返回 HTTP 状态码（通常是 200 或 206）。

    调用方应同时检查状态码与目标文件是否确实落盘。
    网络异常向上抛出，由调用方给出可操作的提示。
    """
    out = Path(out)
    part = out.with_name(out.name + ".part")
    try:
        try:
            resp = _open(url, byte_range, timeout)
        except urllib.error.HTTPError as e:
            return e.code
        with resp:
            status = getattr(resp, "status", None) or resp.getcode()
            with part.open("wb") as f:
                while True:
                    chunk = resp.read(1 << 20)   # 1 MB
                    if not chunk:
                        break
                    f.write(chunk)
        os.replace(part, out)
        return int(status)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def probe_size(url: str, timeout: int = 30) -> int:
    """GET + Range: bytes=0-0，从 Content-Range 读出远端文件总大小；失败返回 0。"""
    try:
        with _open(url, "0-0", timeout) as resp:
            cr = resp.headers.get("Content-Range") or ""
            if "/" in cr:
                return int(cr.rsplit("/", 1)[1].strip())
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else 0
    except Exception:
        return 0
