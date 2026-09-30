"""PDF 文本抽取 + 中文友好的递归字符分块。

接口约定：
    chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]
    chunk_size / overlap 均以「字符」计（不是 token）。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union


# 分隔符按「优先保留大颗粒边界」的顺序试
_SEPS = ("\n\n", "\n", "。", "！", "？", "；", "…", "……",
         ".", "!", "?", ";", "，", ",", " ")


def extract_pdf_text(pdf_path: Union[str, Path]) -> str:
    """从 PDF 抽取纯文本（按页拼接）。

    只用 pypdf；不依赖版式，所以抽出来可能是碎片，交给 chunk_text 做归并。
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("需要 pypdf：pip install pypdf>=4.0.0") from e

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在：{pdf_path}")

    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            parts.append(txt)
    return "\n".join(parts)


def _best_cut(window: str, min_keep: int) -> int:
    """在 window 中找 >= min_keep 的最后一个自然断点（返回断点后一位下标）。

    找不到返回 -1，调用方就按 chunk_size 硬切。
    """
    best = -1
    for sep in _SEPS:
        pos = window.rfind(sep)
        if pos < 0:
            continue
        end = pos + len(sep)
        if end >= min_keep and end > best:
            best = end
    return best


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """按字符切分文本，尽量在段落/句子边界处断开。

    - chunk_size / overlap 以字符计
    - 每块长度目标 chunk_size，切点至少保留 60% * chunk_size，避免碎块
    - overlap 太大会退化为「步长过小」，这里自动裁剪到 chunk_size // 4
    """
    if not isinstance(text, str):
        text = str(text)
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    overlap = max(0, int(overlap))
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    n = len(text)
    if n == 0:
        return []

    # 保证切点不要太靠前，否则平均长度会跌破 chunk_size * 0.5
    min_keep = max(1, int(chunk_size * 0.6))

    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            cut = _best_cut(text[start:end], min_keep)
            if cut > 0:
                end = start + cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        # 下一块起点：留 overlap；但要严格前进，防死循环
        nxt = end - overlap
        if nxt <= start:
            nxt = end
        start = nxt

    return chunks