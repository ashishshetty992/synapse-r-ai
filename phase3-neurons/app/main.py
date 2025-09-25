from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import os, json

from .models import (
    BuildIEMRequest, BuildIEMResponse,
    IntentRequest, IntentEncoding,
    MatchRequest, MatchResponse,
    UEM
)
from .iem import IEMIndex, build_iem_from_uem
from .intent import encode_intent_to_vocab
from .synapse import match_candidates

app = FastAPI(title="SYNAPSE-R Phase 3 Neurons", version="0.1.0")

UEM_PATH = os.environ.get("UEM_PATH", os.path.join(os.getcwd(), "uem.json"))
IEM_PATH = os.environ.get("IEM_PATH", os.path.join(os.getcwd(), "iem.json"))

IEM: Optional[IEMIndex] = None  # hot state


def _load_uem_from_path(path: str) -> UEM:
    with open(path, "r") as f:
        data = json.load(f)
    return UEM.model_validate(data)


def _save_json(path: str, data: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


@app.on_event("startup")
def boot():
    global IEM
    # Try to load/build IEM on startup (best effort)
    try:
        if os.path.exists(IEM_PATH):
            IEM = IEMIndex.load(IEM_PATH)
        elif os.path.exists(UEM_PATH):
            uem = _load_uem_from_path(UEM_PATH)
            iem = build_iem_from_uem(uem, dim=256)
            IEMIndex.save_json(IEM_PATH, iem.to_json())
            IEM = iem
    except Exception as e:
        # Non-fatal: service can still accept /iem/build
        print("Startup warning:", e)


@app.get("/healthz")
def healthz():
    return {"ok": True, "has_iem": IEM is not None}


@app.post("/iem/build", response_model=BuildIEMResponse)
def iem_build(req: BuildIEMRequest = Body(...)):
    """
    Build IEM from:
    - inline `uem` (preferred for dynamic), or
    - `uemPath` file, else
    - default UEM_PATH
    """
    global IEM
    if req.uem is not None:
        uem = req.uem
    else:
        path = req.uemPath or UEM_PATH
        uem = _load_uem_from_path(path)

    dim = req.dim or 256
    iem = build_iem_from_uem(uem, dim=dim)
    IEMIndex.save_json(req.iemPath or IEM_PATH, iem.to_json())
    IEM = iem
    return BuildIEMResponse(
        ok=True,
        dim=iem.dim,
        fieldCount=len(iem.fields),
        savedTo=req.iemPath or IEM_PATH,
    )


@app.post("/intent/encode", response_model=IntentEncoding)
def intent_encode(req: IntentRequest = Body(...)):
    """
    Encodes the intent. If an in-memory IEM is present, returns vector aligned
    to IEM vocab; else builds a free vocab (still deterministic).
    """
    if IEM is not None:
        vec = encode_intent_to_vocab(req.intent, IEM.vocab)
        return IntentEncoding(version="intent/0.2", dim=len(IEM.vocab), vec=vec, vocab=IEM.vocab)
    else:
        # free vocab mode
        vec, vocab = encode_intent_to_vocab(req.intent, None, return_vocab=True)
        return IntentEncoding(version="intent/0.2", dim=len(vocab), vec=vec, vocab=vocab)


@app.post("/synapse/match", response_model=MatchResponse)
def synapse_match(req: MatchRequest = Body(...)):
    """
    Runs matcher over current IEM using:
      - provided intent object (encoded to IEM vocab), or
      - provided raw vec aligned to IEM vocab (same length).
    """
    if IEM is None:
        return JSONResponse(
            {"ok": False, "error": "IEM not loaded. Call /iem/build first."},
            status_code=400,
        )

    if req.intentVec is not None:
        # user guarantees alignment to current IEM vocab
        intent_vec = req.intentVec
        if len(intent_vec) != len(IEM.vocab):
            return JSONResponse(
                {"ok": False, "error": "intentVec dimension does not match current IEM vocab"},
                status_code=400,
            )
    else:
        intent_vec = encode_intent_to_vocab(req.intent or {}, IEM.vocab)

    out = match_candidates(
        iem=IEM,
        intent_vec=intent_vec,
        top_k_fields=req.topKFields or 8,
        top_k_entities=req.topKEntities or 5,
        role_alpha=req.roleAlpha or 0.12,
    )
    return out