import os, json, time, gzip, threading, glob
from typing import Optional, Dict, Any
from datetime import datetime

_DEFAULT_PATH = os.environ.get("NEURONS_DECISION_LOG", "./logs/decisions.jsonl.gz")
os.makedirs(os.path.dirname(_DEFAULT_PATH), exist_ok=True)
_lock = threading.Lock()

def _today_path(base_path: str) -> str:
    """Return a rotated path like decisions.jsonl.2025-11-03.gz"""
    root, ext = os.path.splitext(base_path)
    date_tag = datetime.utcnow().strftime("%Y-%m-%d")
    return f"{root}.{date_tag}{ext}"

def _open_append(path: str):
    return gzip.open(path, mode="ab")

def log_decision(event: str, rid: str, payload: Dict[str, Any],
                 path: Optional[str] = None, enabled: Optional[bool] = None):
    if enabled is False:
        return
    base = path or _DEFAULT_PATH
    p = _today_path(base)
    rec = {"ts": time.time(), "event": event, "rid": rid, **payload}
    data = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")
    with _lock:
        with _open_append(p) as f:
            f.write(data)

def _latest_shard(base_path: str) -> str | None:
    root, ext = os.path.splitext(base_path)
    candidates = sorted(glob.glob(f"{root}.*{ext}"))
    return candidates[-1] if candidates else (base_path if os.path.exists(base_path) else None)

def read_tail(n: int = 200, path: Optional[str] = None) -> list[dict]:
    base = path or _DEFAULT_PATH
    p = _latest_shard(base)
    out = []
    if not p or not os.path.exists(p):
        return out
    with gzip.open(p, "rb") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out[-n:]