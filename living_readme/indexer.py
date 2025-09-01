from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from rich.progress import track
from .config import load_settings


@dataclass
class Chunk:
    id: str
    path: str
    start: int
    end: int
    text: str




def iter_files(root: str, include_exts: List[str], exclude_dirs: List[str]):
    root_path = Path(root)
    for p in root_path.rglob("*"):
        if p.is_dir():
            if p.name in exclude_dirs:
                # Skip entire subtree
                dirs = [] # not used here, rglob can't prune, so check below
            continue
        if p.suffix.lower() in include_exts and not any(part in exclude_dirs for part in p.parts):
            yield p




def chunk_text(text: str, size: int, overlap: int) -> List[Tuple[int, int, str]]:
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + size, n)
        chunks.append((i, end, text[i:end]))
        if end == n:
            break
        i = end - overlap
    return chunks




def build_chunks() -> List[Chunk]:
    s = load_settings()
    chunks: List[Chunk] = []
    for path in iter_files(s.REPO_ROOT, s.INCLUDE_EXTS, s.EXCLUDE_DIRS):
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for idx, (st, en, ch) in enumerate(chunk_text(txt, s.CHUNK_SIZE, s.CHUNK_OVERLAP)):
            cid = f"{path}:{idx}"
            chunks.append(Chunk(id=cid, path=str(path), start=st, end=en, text=ch))
    return chunks