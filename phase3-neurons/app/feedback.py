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
    if len(str(rec.intent)) > 32_000:
        raise HTTPException(status_code=413, detail="payload too large")

    # Derive iem hash if not provided
    iem_hash = rec.iemHash or sha256_file(IEM_PATH)

    # Scrub PII if flagged
    intent = rec.intent.copy()
    if rec.sensitive:
        for k in list(intent.keys()):
            if k in SENSITIVE_KEYS:
                intent[k] = "<redacted>"

    row = {
        'id': str(uuid.uuid4()),
        'ts': int(time.time()*1000),
        'tenant': rec.tenant,
        'intent': intent,
        'chosenTarget': rec.chosenTarget,
        'otherSlots': rec.otherSlots,
        'iemHash': iem_hash,
        'latencyMs': rec.latencyMs,
        'ua': user_agent,
        'reqId': request.headers.get('x-request-id')
    }
    path = os.path.join(DATA_DIR, 'feedback.jsonl')
    append_jsonl(path, row)
    return FeedbackRecordOut(ok=True, id=row['id'])