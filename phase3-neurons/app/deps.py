from __future__ import annotations
import os, json, hashlib
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
GOLDEN_DIR = os.path.join(ROOT, 'golden')
CKPT_DIR = os.path.join(ROOT, 'checkpoints')
RUNTIME_DIR = os.path.join(ROOT, 'runtime')
ACTIVE_PTR = os.path.join(RUNTIME_DIR, 'active.json')
IEM_PATH = os.path.join(ROOT, 'iem.json')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GOLDEN_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RUNTIME_DIR, exist_ok=True)

CLAMPS = {
    'alias_alpha': (0.25, 0.60),
    'shape_alpha': (0.10, 0.60),
    'shape_beta':  (0.10, 0.60),
    'metric_alpha':(0.10, 0.70),
    'role_alpha_default':(0.10, 0.60),
    'pathScoring.idPrior':(0.0, 0.5),
    'pathScoring.fkBonus':(0.0, 0.5),
    'pathScoring.lengthPriorBase':(0.5, 1.0),
    'pathScoring.cosineWeight':(0.0, 1.0),
}

SENSITIVE_KEYS = {'freeText','rawQuery','email','phone'}


def clamp(name: str, val: float) -> float:
    lo, hi = CLAMPS.get(name, (None, None))
    if lo is None: return val
    return max(lo, min(hi, val))


def sha256_file(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None


def read_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)