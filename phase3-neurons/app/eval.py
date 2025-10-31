# phase3-neurons/app/eval.py
from __future__ import annotations
from fastapi import APIRouter
import os, json
from .deps import DATA_DIR, GOLDEN_DIR, ACTIVE_PTR
from .utils_jsonl import read_jsonl
from .models import EvalOut

router = APIRouter(prefix="/trainer", tags=["trainer"])

@router.get('/last_eval', response_model=EvalOut)
async def last_eval():
    n_feedback = len(read_jsonl(os.path.join(DATA_DIR, 'feedback.jsonl')))
    n_golden = 0
    try:
        for fn in os.listdir(GOLDEN_DIR):
            if fn.endswith('.json'):
                n_golden += 1
    except FileNotFoundError:
        pass

    ckpt = None
    try:
        with open(ACTIVE_PTR, 'r') as f:
            ckpt = json.load(f).get('checkpoint')
    except FileNotFoundError:
        ckpt = None

    # Placeholder metrics (will be computed once trainer integrates)
    return EvalOut(
        top1_acc=0.0, mrr=0.0, slot_f1=0.0,
        aplus_mean=0.0, pathscore_mean=0.0,
        n_golden=n_golden, n_feedback=n_feedback, ckpt=ckpt
    )