import os, json, time, gzip, threading
from typing import Optional, Dict, Any

_DEFAULT_PATH = os.environ.get("NEURONS_DECISION_LOG", "./logs/decisions.jsonl.gz")
os.makedirs(os.path.dirname(_DEFAULT_PATH), exist_ok=True)
_lock = threading.Lock()

def _open_append(path: str):
    # Append a new gzip member (valid concatenated gzip stream)
    return gzip.open(path, mode="ab")

def log_decision(event: str, rid: str, payload: Dict[str, Any],
                 path: Optional[str] = None, enabled: Optional[bool] = None):
    if enabled is False:
        return
    p = path or _DEFAULT_PATH
    rec = {"ts": time.time(), "event": event, "rid": rid, **payload}
    data = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")
    with _lock:
        with _open_append(p) as f:
            f.write(data)

def read_tail(n: int = 200, path: Optional[str] = None) -> list[dict]:
    p = path or _DEFAULT_PATH
    out = []
    if not os.path.exists(p):
        return out
    with gzip.open(p, "rb") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out[-n:]

