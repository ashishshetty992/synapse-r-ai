# phase3-neurons/app/feedback.py
from __future__ import annotations
from fastapi import APIRouter, Header, Request, HTTPException
from .models import FeedbackRecordIn, FeedbackRecordOut
from .utils_jsonl import append_jsonl
from .deps import DATA_DIR, SENSITIVE_KEYS, sha256_file, IEM_PATH
import os, time, uuid

router = APIRouter(prefix="/feedback", tags=["feedback"])

@router.post('/record', response_model=FeedbackRecordOut)
async def record_feedback(rec: FeedbackRecordIn, request: Request,
                          user_agent: str | None = Header(default=None)):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if len(str(rec.intent)) > 32_000:
        from fastapi.responses import JSONResponse
        return JSONResponse({"ok": False, "error": "payload too large", "requestId": rid}, status_code=413)

    # Normalize tenant
    tenant = (rec.tenant or "default").strip()[:128]
    
    # Scrub PII from intent
    intent = rec.intent.copy() if isinstance(rec.intent, dict) else rec.intent
    if rec.sensitive:
        if isinstance(intent, dict):
            for k in list(intent.keys()):
                if k in SENSITIVE_KEYS:
                    intent[k] = "<redacted>"
        intent = str(intent)[:5000]
    else:
        intent = str(intent)[:5000]
    
    # Scrub PII from otherSlots
    def _scrub_slots(d):
        if not isinstance(d, dict): 
            return None
        return {k: v for k, v in d.items() if k not in SENSITIVE_KEYS}
    
    otherSlots = _scrub_slots(rec.otherSlots) or {}
    
    # Derive iem hash if not provided
    iem_hash = rec.iemHash
    if not iem_hash:
        iem_hash = sha256_file(IEM_PATH) if os.path.exists(IEM_PATH) else None
    
    # Validate and clamp latency
    lat = rec.latencyMs
    try:
        lat = max(0, min(int(lat), 10_000)) if lat is not None else None
    except Exception:
        lat = None

    row = {
        'id': str(uuid.uuid4()),
        'ts': int(time.time()*1000),
        'tenant': tenant,
        'intent': intent,
        'chosenTarget': rec.chosenTarget,
        'otherSlots': otherSlots,
        'iemHash': iem_hash,
        'latencyMs': lat,
        'ua': user_agent,
        'reqId': request.headers.get('x-request-id')
    }
    path = os.path.join(DATA_DIR, 'feedback.jsonl')
    append_jsonl(path, row)
    return FeedbackRecordOut(ok=True, id=row['id'], requestId=rid)