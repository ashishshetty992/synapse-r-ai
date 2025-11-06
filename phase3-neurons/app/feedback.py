from __future__ import annotations
from fastapi import APIRouter, Request
import os, json, time, uuid
from .deps import DATA_DIR
from .models import FeedbackRecordIn, FeedbackRecordOut

router = APIRouter(prefix="/feedback", tags=["feedback"])

def _append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")

@router.post("/record", response_model=FeedbackRecordOut)
async def record(inp: FeedbackRecordIn, request: Request = None):
    rid = getattr(request.state, "request_id", "-")
    row = {
        "ts": int(time.time()*1000),
        "tenant": inp.tenant or (getattr(request.state, "tenant_id", None) or "default"),
        "intent": (None if inp.sensitive else inp.intent),
        "chosenTarget": inp.chosenTarget,
        "otherSlots": (None if inp.sensitive else inp.otherSlots),
        "iemHash": inp.iemHash,
        "latencyMs": inp.latencyMs,
        "ua": (request.headers.get("user-agent") if request else None),
        "rid": rid
    }
    fid = uuid.uuid4().hex
    path = os.path.join(DATA_DIR, "feedback.jsonl")
    _append_jsonl(path, {"id": fid, **row})
    return FeedbackRecordOut(ok=True, id=fid, requestId=rid)
