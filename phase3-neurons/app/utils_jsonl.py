# phase3-neurons/app/utils_jsonl.py
from __future__ import annotations
import json, os
from typing import Iterable


def append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # best-effort atomic append on POSIX: write temp then append
    tmp = f"{path}.tmp"
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(line)
    with open(tmp, "rb") as src, open(path, "ab") as dst:
        dst.write(src.read())
    try:
        os.remove(tmp)
    except Exception:
        pass


def read_jsonl(path: str, limit: int|None=None) -> list[dict]:
    out = []
    try:
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if line.strip():
                    out.append(json.loads(line))
                if limit is not None and i+1 >= limit:
                    break
    except FileNotFoundError:
        return []
    return out