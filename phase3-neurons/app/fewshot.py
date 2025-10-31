# phase3-neurons/app/fewshot.py
from __future__ import annotations
from fastapi import APIRouter, Query
import json, os
from .deps import ACTIVE_PTR
from .models import FewshotShowOut

router = APIRouter(prefix="/fewshot", tags=["fewshot"])

@router.get("/show", response_model=FewshotShowOut)
async def fewshot_show(tenant: str = Query(default="default")):
    try:
        with open(ACTIVE_PTR, "r") as f:
            ptr = json.load(f)
    except FileNotFoundError:
        return FewshotShowOut(tenant=tenant, merged={})

    fs_path = ptr.get("fewshot")
    if not fs_path or not os.path.exists(fs_path):
        return FewshotShowOut(tenant=tenant, merged={})

    with open(fs_path, "r") as f:
        fs = json.load(f)

    merged = {"hints": [], "aliases": {}}
    g = fs.get("global", {})
    merged["hints"] += g.get("hints", [])
    merged["aliases"].update(g.get("aliases", {}))

    t = fs.get("tenants", {}).get(tenant, {})
    merged["hints"] += t.get("hints", [])
    merged["aliases"].update(t.get("aliases", {}))

    return FewshotShowOut(tenant=tenant, merged=merged)