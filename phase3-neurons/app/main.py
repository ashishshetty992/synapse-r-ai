from fastapi import FastAPI, Body, Request, Query
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any, List
import os, json, uuid, logging, time
from collections import deque, defaultdict

from .feedback import router as feedback_router
from .eval import router as eval_router
from .trainer import router as trainer_router
from .fewshot import router as fewshot_router
from .config_api import router as config_router

from .models import (
    BuildIEMRequest, BuildIEMResponse,
    IntentRequest, IntentEncoding,
    MatchRequest, MatchResponse, ScoredField, ScoredEntity,
    VerifyIEMResponse, RoleBoostStat, DriftAlert, ClusterQuality as CQ,
    NLRequest, IntentEncodingNL,
    FillRequest, FillResponse, ConflictNote, AlignPlus,
    PathsRequest, PathsResponse, PathCandidate, PathEdge,
    ExplainRequest, ExplainResponse, ExplainItem, CandidateScore,
    GenerateRequest, GenerateExampleResponse, GenIQL, CoverageStat,
    UEM
)
from .iem import IEMIndex, build_iem_from_uem, compute_role_boost, compute_alias_stability, cluster_quality
from .intent import encode_intent_to_vocab, encode_intent_nl
from .synapse import match_candidates, compute_alignplus, _parse_entity_field, _build_join_graph, _bfs_paths, _path_score, score_target_context
from .config import load_role_cfg, load_shaping_cfg, load_entity_cfg, load_synonyms, load_time_cfg
from .generator import generate_all, write_examples, write_coverage

# Structured logging setup
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(request_id)s] %(message)s",
    level=logging.INFO
)

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True

# -------- Structured JSON logging + request-id + auth + rate limit --------

def log_json(level: str, rid: str, event: str, **kwargs):
    payload = {"ts": time.time(), "level": level, "rid": rid, "event": event, **kwargs}
    logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(payload), extra={"request_id": rid})

# Response helpers
def ok(rid: str, payload: dict) -> dict:
    return {"ok": True, "requestId": rid, **payload}

def err(status: int, rid: str, msg: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": msg, "requestId": rid}, status_code=status)

_rate_buckets: dict[str, deque] = defaultdict(deque)  # per-IP sliding window

app = FastAPI(title="SYNAPSE-R Phase 3 Neurons", version="0.1.0")

# Router includes
app.include_router(feedback_router)
app.include_router(eval_router)
app.include_router(trainer_router)
app.include_router(fewshot_router)
app.include_router(config_router)

logger = logging.getLogger("neurons")
logger.addFilter(RequestIdFilter())

# Request-ID middleware
@app.middleware("http")
async def guardrails_mw(request: Request, call_next):
    # request-id
    rid = str(uuid.uuid4())
    request.state.request_id = rid

    # client ip (trusting uvicorn's client or X-Forwarded-For if present)
    if TRUST_PROXY and request.headers.get("x-forwarded-for"):
        ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    
    # user agent
    ua = request.headers.get("user-agent")

    # tenant id (header → query → None)
    tenant = request.headers.get("x-tenant-id") or request.query_params.get("tenant")
    request.state.tenant_id = tenant

    # Load tenant configs and apply to synapse (overrides globals on every request)
    try:
        from . import synapse
        rc = load_role_cfg(tenant)
        sc = load_shaping_cfg(tenant)
        ec = load_entity_cfg(tenant)

        log_json("info", rid, "cfg.load.start", tenant=tenant, 
                role_cfg_keys=list(rc.keys()) if rc else [],
                shaping_cfg_keys=list(sc.keys()) if sc else [],
                entity_cfg_keys=list(ec.keys()) if ec else [])

        if isinstance(rc.get("keywords"), dict):
            synapse.set_role_config(rc["keywords"])
            log_json("info", rid, "cfg.role.applied", keyword_count=len(rc["keywords"]))

        if isinstance(sc.get("weights"), dict):
            synapse.set_shaping_weights(sc["weights"])
            # NEW: honor role_alpha_default if trainer wrote it under weights
            if sc["weights"].get("role_alpha_default") is not None:
                synapse.set_role_alpha_default(float(sc["weights"]["role_alpha_default"]))
            log_json("info", rid, "cfg.shaping.applied", weight_count=len(sc["weights"]))

        # entity blend + centroid gamma + roleAlpha default
        eb = (ec.get("entity_blend") or {})
        synapse.set_entity_blend(
            entity_weight=float(eb.get("entityWeight", 0.70)),
            field_weight=float(eb.get("fieldWeight", 0.30)),
            entity_alias_alpha=float(eb.get("entityAliasAlpha", 0.15)),
        )
        synapse.set_centroid_gamma(float(ec.get("centroid_gamma", 0.60)))
        synapse.set_role_alpha_default(float(ec.get("role_alpha", 0.12)))
        
        # NEW: pass fill section (if any)
        if isinstance(sc.get("fill"), dict):
            synapse.set_fill_config(sc["fill"])
        
        # NEW: path scoring knobs
        if isinstance(sc.get("pathScoring"), dict):
            synapse.set_path_scoring(sc["pathScoring"])
        
        # NEW: logging config
        if isinstance(sc.get("logging"), dict):
            synapse.set_logging_cfg(sc["logging"])
        
        log_json("info", rid, "cfg.entity.applied", 
                entity_weight=eb.get("entityWeight", 0.70),
                field_weight=eb.get("fieldWeight", 0.30),
                entity_alias_alpha=eb.get("entityAliasAlpha", 0.15),
                centroid_gamma=ec.get("centroid_gamma", 0.60),
                role_alpha=ec.get("role_alpha", 0.12))
                
    except Exception as e:
        log_json("warning", rid, "cfg.load.error", error=str(e))

    # auth (if API_KEY configured)
    if API_KEY:
        given = request.headers.get("x-api-key")
        if given != API_KEY:
            log_json("warning", rid, "auth.denied", path=str(request.url), method=request.method, ip=ip)
            return JSONResponse({"ok": False, "error": "unauthorized", "requestId": rid}, status_code=401)

    # simple sliding-window rate limit per IP
    now = time.time()
    bucket = _rate_buckets[ip]
    # drop old entries
    while bucket and now - bucket[0] > RL_WINDOW_SEC:
        bucket.popleft()
    if len(bucket) >= RL_MAX:
        reset = RL_WINDOW_SEC - int(now - bucket[0])
        headers = {
            "x-request-id": rid,
            "x-ratelimit-limit": str(RL_MAX),
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": str(reset),
        }
        log_json("warning", rid, "ratelimit.block", path=str(request.url), method=request.method, ip=ip, limit=RL_MAX, window=RL_WINDOW_SEC)
        return err(429, rid, "rate_limited")
    bucket.append(now)

    # timing + pass through
    t0 = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        # log server error
        log_json("error", rid, "unhandled.error", path=str(request.url), method=request.method, ip=ip, error=str(e))
        raise
    duration_ms = int((time.time() - t0) * 1000)
    # ratelimit headers
    remaining = max(0, RL_MAX - len(bucket))
    response.headers["x-request-id"] = rid
    response.headers["x-ratelimit-limit"] = str(RL_MAX)
    response.headers["x-ratelimit-remaining"] = str(remaining)
    # access log
    log_json("info", rid, "http.access", path=str(request.url), method=request.method, status=getattr(response, "status_code", 200), ip=ip, ua=ua, dur_ms=duration_ms, tenant=getattr(request.state, "tenant_id", None))
    return response

UEM_PATH = os.environ.get("UEM_PATH", os.path.join(os.getcwd(), "uem.json"))
IEM_PATH = os.environ.get("IEM_PATH", os.path.join(os.getcwd(), "iem.json"))

ROLE_CFG_PATH = os.environ.get("NEURONS_ROLE_CFG")  # optional JSON file
SHAPING_CFG_PATH = os.environ.get("NEURONS_SHAPING_CFG")  # optional JSON file

API_KEY = os.environ.get("NEURONS_API_KEY")  # set to enable auth; if None -> auth disabled
RL_MAX = int(os.environ.get("NEURONS_RL_MAX", "60"))         # requests per window
RL_WINDOW_SEC = int(os.environ.get("NEURONS_RL_WINDOW", "60"))  # window size (s)
TRUST_PROXY = os.environ.get("NEURONS_TRUST_PROXY", "0") == "1"
OUT_BASE = os.path.abspath(os.environ.get("NEURONS_OUT_BASE", os.getcwd()))

IEM: Optional[IEMIndex] = None  # hot state
COVERAGE_CACHE: Optional[List[Dict[str, Any]]] = None  # coverage from last /generate


def _load_uem_from_path(path: str) -> UEM:
    with open(path, "r") as f:
        data = json.load(f)
    return UEM.model_validate(data)


def _save_json(path: str, data: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json_if(path: Optional[str]) -> Optional[dict]:
    if not path: return None
    if not os.path.exists(path): return None
    with open(path, "r") as f:
        return json.load(f)


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
        logger.warning(f"Startup warning: {e}")
    
    # apply optional configs
    from .synapse import set_role_config, set_shaping_weights
    role_cfg = _load_json_if(ROLE_CFG_PATH)
    if role_cfg and isinstance(role_cfg.get("keywords"), dict):
        set_role_config(role_cfg["keywords"])
    shaping_cfg = _load_json_if(SHAPING_CFG_PATH)
    if shaping_cfg and isinstance(shaping_cfg.get("weights"), dict):
        set_shaping_weights(shaping_cfg["weights"])


@app.get("/healthz")
def healthz():
    return {"ok": True, "has_iem": IEM is not None}


@app.post("/iem/build", response_model=BuildIEMResponse)
def iem_build(req: BuildIEMRequest = Body(...), request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    try:
        global IEM
        if req.dim is not None and (req.dim < 32 or req.dim > 2048):
            return err(400, rid, "dim out of range [32,2048]")

        # load UEM
        if req.uem is not None:
            uem = req.uem
        else:
            path = req.uemPath or UEM_PATH
            if not os.path.exists(path):
                return err(400, rid, f"UEM file not found: {path}")
            uem = _load_uem_from_path(path)

        dim = req.dim or 256
        iem = build_iem_from_uem(uem, dim=dim)

        # backup existing IEM for drift detection
        out_path = req.iemPath or IEM_PATH
        try:
            if os.path.exists(out_path):
                os.replace(out_path, out_path + ".bak")
                log_json("info", rid, "iem.backup", path=out_path + ".bak")
        except Exception as be:
            log_json("warning", rid, "iem.backup.error", error=str(be))

        IEMIndex.save_json(out_path, iem.to_json())
        IEM = iem

        log_json("info", rid, "iem.build", dim=iem.dim, fields=len(iem.fields))
        return BuildIEMResponse(**ok(rid, {"dim": iem.dim, "fieldCount": len(iem.fields), "savedTo": out_path}))
    except Exception as e:
        log_json("error", rid, "iem.build.error", error=str(e))
        return err(500, rid, str(e))


@app.post("/intent/encode", response_model=IntentEncoding)
def intent_encode(req: IntentRequest = Body(...), request: Request = None):
    """
    Encodes the intent. If an in-memory IEM is present, returns vector aligned
    to IEM vocab; else builds a free vocab (still deterministic).
    """
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is not None:
        vec = encode_intent_to_vocab(req.intent, IEM.vocab)
        log_json("info", rid, "intent.encode", dim=len(vec))
        return IntentEncoding(version="intent/0.2", dim=len(IEM.vocab), vec=vec, vocab=IEM.vocab, requestId=rid)
    else:
        # free vocab mode
        vec, vocab = encode_intent_to_vocab(req.intent, None, return_vocab=True)
        log_json("info", rid, "intent.encode", dim=len(vec))
        return IntentEncoding(version="intent/0.2", dim=len(vocab), vec=vec, vocab=vocab, requestId=rid)


@app.post("/intent/encode_nl", response_model=IntentEncodingNL)
def intent_encode_nl(req: NLRequest = Body(...), request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")
    try:
        synonyms = load_synonyms()
        tenant = getattr(request.state, "tenant_id", None)
        time_cfg = load_time_cfg(tenant)  # ← config-driven, tenant-aware
        out = encode_intent_nl(req.text, IEM, synonyms, topK=(req.topK or 8), time_cfg=time_cfg)
        log_json("info", rid, "intent.encode_nl", dim=out["dim"])
        return IntentEncodingNL(
            version=out["version"],
            dim=out["dim"],
            vec=out["vec"],
            vocab=IEM.vocab,
            debug={
                "topAliasHits": out["topAliasHits"],
                "blend": out["blend"],
                "text": req.text,
                **out.get("debug", {}),
            },
            requestId=rid
        )
    except Exception as e:
        log_json("error", rid, "intent.encode_nl.error", error=str(e))
        return err(500, rid, str(e))


@app.post("/synapse/match", response_model=MatchResponse)
def synapse_match(req: MatchRequest = Body(...), request: Request = None):
    """
    Runs matcher over current IEM using:
      - provided intent object (encoded to IEM vocab), or
      - provided raw vec aligned to IEM vocab (same length).
    """
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")

    if req.intentVec is not None:
        # user guarantees alignment to current IEM vocab
        intent_vec = req.intentVec
        if len(intent_vec) != len(IEM.vocab):
            return err(400, rid, "intentVec dimension does not match current IEM vocab")
    else:
        intent_vec = encode_intent_to_vocab(req.intent or {}, IEM.vocab)

    out = match_candidates(
        iem=IEM,
        intent_vec=intent_vec,
        top_k_fields=req.topKFields or 8,
        top_k_entities=req.topKEntities or 5,
        role_alpha=req.roleAlpha,   # <-- let it be None so synapse uses _CONFIG["role_alpha_default"]
        intent_obj=(req.intent or {}),
        # + NEW: pass shaping parameters for A/B tuning
        alias_alpha=req.aliasAlpha,
        shape_alpha=req.shapeAlpha,
        shape_beta=req.shapeBeta,
        metric_alpha=req.metricAlpha,
    )
    out.debug["tenantId"] = getattr(request.state, "tenant_id", None)
    out.debug["requestId"] = rid
    log_json("info", rid, "synapse.match", topFields=len(out.topFields))
    return out


@app.get("/iem/verify", response_model=VerifyIEMResponse)
def iem_verify(request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded")

    # RoleBoost per-field
    role_boost_stats: List[RoleBoostStat] = []
    for f in IEM.fields:
        top_role = max(f.role.items(), key=lambda kv: kv[1])[0] if f.role else None
        score = f.roleBoost if f.roleBoost is not None else compute_role_boost(f)
        role_boost_stats.append(RoleBoostStat(
            entity=f.entity, field=f.name, roleTop=top_role, boostScore=round(float(score), 4)
        ))

    # Drift vs .bak (if present)
    drift_alerts: List[DriftAlert] = []
    prev_path = (IEM_PATH + ".bak")
    if os.path.exists(prev_path):
        try:
            prev = IEMIndex.load(prev_path)
            prev_map = {(p.entity, p.name): p for p in prev.fields}
            for f in IEM.fields:
                p = prev_map.get((f.entity, f.name))
                if p:
                    alias_diff, vec_diff = compute_alias_stability(f.aliases, p.aliases, f.vec, p.vec)
                    drift_alerts.append(DriftAlert(
                        field=f"{f.entity}.{f.name}",
                        aliasChange=round(alias_diff, 4),
                        vecDrift=round(vec_diff, 4),
                        warning="high drift" if (alias_diff > 0.5 or vec_diff > 0.5) else None
                    ))
        except Exception as e:
            log_json("warning", rid, "iem.verify.drift.error", error=str(e))

    # Cluster quality by role
    clusters: List[CQ] = []
    for role in ["id", "timestamp", "money", "geo", "category", "text", "quantity"]:
        score = cluster_quality(IEM.fields, role)
        if score is not None:
            count = sum(1 for f in IEM.fields if role in f.role)
            clusters.append(CQ(role=role, avgSim=round(float(score), 4), count=count))

    warnings = [f"Low cluster cohesion for role {c.role}" for c in clusters if c.avgSim < 0.5]

    log_json("info", rid, "iem.verify", rbCount=len(role_boost_stats), driftCount=len(drift_alerts))
    return VerifyIEMResponse(
        ok=True,
        roleBoost=role_boost_stats,
        drift=drift_alerts,
        clusters=clusters,
        warnings=warnings,
        requestId=rid
    )


def _role_of(field) -> str:
    if not field.role: return "unknown"
    return max(field.role.items(), key=lambda kv: kv[1])[0]

def _is_money_op(metric: Dict[str, Any] | None) -> bool:
    if not metric: return False
    op = (metric or {}).get("op","").lower()
    return op in {"sum","avg","mean","median"}

def _apply_preferences(
    cands: List[Dict[str, Any]],
    prefer_roles: Optional[List[str]] = None,
    prefer_targets: Optional[List[str]] = None,
    prefer_targets_boost: Optional[Dict[str, float]] = None,
    alias_targets: Optional[List[str]] = None,
    ask: Optional[str] = None,
    money_op: bool = False,
    fill_cfg: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Apply weighted preferences to candidate fields.
    Returns list of dicts with updated scores.
    """
    fill_cfg = fill_cfg or {}
    role_bonus = float(fill_cfg.get("preferBonus", 0.08))
    target_bonus = 0.12  # a bit stronger than before
    alias_bonus = 0.30   # strong: when NL explicitly says city
    granularity_bonus_city = 0.12  # city > country when ask=top_k
    money_penalty = float(fill_cfg.get("moneyPivotPenalty", 0.08))
    unknown_penalty = float(fill_cfg.get("unknownPenalty", 0.1))
    
    # fast lookup
    idx = {(c["entity"], c["name"]): i for i, c in enumerate(cands)}
    
    def bump(field: str, delta: float):
        """Apply delta boost to a field if it exists."""
        if "." in field:
            ent, name = field.split(".", 1)
            j = idx.get((ent, name))
            if j is not None:
                cands[j]["score"] += delta
    
    # 1) role-level nudges
    if prefer_roles:
        pr = set(prefer_roles)
        for c in cands:
            if c.get("roleTop") in pr:
                c["score"] += role_bonus
    
    # 2) list-based targets (uniform)
    if prefer_targets:
        for t in prefer_targets:
            bump(t, target_bonus)
    
    # 3) weighted boosts
    if prefer_targets_boost:
        for t, delta in prefer_targets_boost.items():
            bump(t, float(delta))
    
    # 4) alias/context bonus from NL encoder topAliasHits
    if alias_targets:
        for t in alias_targets:
            # Only apply to schema-looking targets (entity.field format)
            if "." in t and t.count(".") == 1:
                bump(t, alias_bonus)
    
    # 5) granularity rule: city beats country for top_k (count/sum)
    if ask == "top_k":
        bump("orders.shipping_city", granularity_bonus_city)
        # no negative penalty to country; just a positive nudge to city
    
    # 6) money pivot penalty (only affects choosing pivot target)
    if money_op:
        for c in cands:
            if c.get("roleTop") == "money":
                c["score"] -= money_penalty
    
    # 7) unknown penalty (downrank unknowns)
    for c in cands:
        if c.get("roleTop") == "unknown":
            c["score"] -= unknown_penalty
    
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands

def _explain_candidate_scores(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate explain output for candidate scores."""
    return [
        {
            "field": f"{c['entity']}.{c['name']}",
            "role": c.get("roleTop", "unknown"),
            "score": round(c["score"], 6)
        }
        for c in cands
    ]

@app.get("/iem/show")
def iem_show(request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded")
    ents = defaultdict(list)
    for f in IEM.fields:
        ents[f.entity].append({
            "name": f.name,
            "roleTop": max(f.role.items(), key=lambda kv: kv[1])[0] if f.role else "unknown",
            "roles": f.role,
            "aliases": f.aliases
        })
    out = []
    for e, fields in ents.items():
        out.append({"entity": e, "fields": fields})
    return ok(rid, {"dim": IEM.dim, "entities": out})

@app.post("/synapse/fill", response_model=FillResponse)
def synapse_fill(req: FillRequest = Body(...), request: Request = None):
    """Slot filling + conflict resolve + AlignPlus™."""
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")
    
    try:
        # Import synapse for _CONFIG access
        from . import synapse
        
        # 0) Optional per-call overrides for fill knobs
        fill_cfg = dict(synapse._CONFIG["fill"])
        if req.fillOverrides:
            for k, v in (req.fillOverrides or {}).items():
                if k in fill_cfg and v is not None:
                    fill_cfg[k] = v

        # 1) Get base ranking from existing matcher (cosine + shaping)
        top_k = max(1, req.topKTargets or 8)
        match_out = match_candidates(
            iem=IEM,
            intent_vec=encode_intent_to_vocab(req.intent or {}, IEM.vocab),
            top_k_fields=max(64, top_k * 4),   # grab generous pool to post-filter
            top_k_entities=10,
            role_alpha=(req.intent.get("roleAlpha") if isinstance(req.intent, dict) else None),
            intent_obj=req.intent or {},
        )

        # 2) Post-process field candidates → apply fill knobs
        money_op = _is_money_op(req.intent.get("metric"))

        raw_fields = match_out.topFields[:]  # ScoredField list
        
        # Convert ScoredField to dict for _apply_preferences
        cands = [{"entity": sf.entity, "name": sf.name, "score": float(sf.score), "roleTop": sf.roleTop or _role_of(sf)} for sf in raw_fields]
        
        # Apply weighted preferences
        adjusted_dicts = _apply_preferences(
            cands,
            prefer_roles=req.preferRoles,
            prefer_targets=req.preferTargets,
            prefer_targets_boost=req.preferTargetsBoost,
            alias_targets=req.aliasTargets,
            ask=req.intent.get("ask") if isinstance(req.intent, dict) else None,
            money_op=money_op,
            fill_cfg=fill_cfg
        )
        
        # Convert back to ScoredField objects
        adjusted = [ScoredField(entity=c["entity"], name=c["name"], score=c["score"], roleTop=c["roleTop"]) for c in adjusted_dicts]
        
        # sort after adjustments (already sorted by _apply_preferences, but ensure)
        adjusted.sort(key=lambda x: x.score, reverse=True)

        # 3) Enforce maxUnknownInTopK clamp
        max_unknown = int(fill_cfg["maxUnknownInTopK"])
        picked, unknown_count = [], 0
        for cand in adjusted:
            if len(picked) >= top_k: break
            if cand.roleTop == "unknown":
                if unknown_count >= max_unknown:
                    continue
                unknown_count += 1
            picked.append(cand)

        # If the clamp made us return fewer than K (e.g., many unknowns), backfill with remaining non-unknowns
        if len(picked) < top_k:
            for cand in adjusted:
                if len(picked) >= top_k: break
                if cand in picked: continue
                if cand.roleTop != "unknown":
                    picked.append(cand)

        # 4) Choose pivot target (first candidate after adjustments)
        pivot = picked[0] if picked else (adjusted[0] if adjusted else None)
        intent_filled = dict(req.intent)
        if pivot:
            intent_filled["target"] = f"{pivot.entity}.{pivot.name}"

        # 5) Conflicts (intelligent resolution)
        from .synapse import resolve_conflicts
        intent_vec = encode_intent_to_vocab(intent_filled, IEM.vocab)
        conflicts_resolved, upgraded_target = resolve_conflicts(
            iem=IEM,
            intent_vec=intent_vec,
            intent_obj=intent_filled,
            raw_fields=raw_fields,
            pivot=pivot,
            match_debug=match_out.debug
        )

        # If resolver suggests a better pivot within same slot-role, upgrade
        if upgraded_target:
            intent_filled["target"] = upgraded_target

        # AlignPlus metrics (coverage/OSR scaffold) – unchanged
        coverage = 1.0 if pivot else 0.0
        osr_from_intent_only = bool(fill_cfg.get("osrFromIntentOnly", True))
        off_schema_rate = 1.0 if (pivot and pivot.roleTop == "unknown") else 0.0
        if not osr_from_intent_only:
            off_schema_rate *= 0.5

        # Abase from raw_fields (unchanged)
        abase = 0.0
        if pivot:
            for sf in raw_fields:
                if sf.entity == pivot.entity and sf.name == pivot.name:
                    abase = float(sf.score); break
        aplus = abase * coverage * (1.0 - off_schema_rate)

        # entities passthrough (unchanged)
        ents = [ScoredEntity(entity=e.entity, score=float(e.score)) for e in match_out.topEntities]

        debug = {
            "topAliasHits": [],
            "tokens": [],
            "mappedTerms": [],
            "rolePrior": match_out.debug.get("rolePrior", {}),
            "matchDebug": match_out.debug,
            "fillConfigUsed": fill_cfg,
            "preferRolesUsed": list(req.preferRoles or []),
            "preferTargetsUsed": req.preferTargets or [],
            "preferTargetsBoostUsed": req.preferTargetsBoost or {},
            "aliasTargetsUsed": req.aliasTargets or [],
        }
        
        # Add explain if requested
        if req.debugExplain:
            debug["targetExplain"] = _explain_candidate_scores(adjusted_dicts)

        return FillResponse(
            ok=True,
            intentFilled=intent_filled,
            targetCandidates=picked,
            entityCandidates=ents,
            conflicts=[ConflictNote(
                slot=c["slot"],
                candidates=c["candidates"],
                resolution=c.get("resolution"),
                why=c.get("why", {}),
                scores=c.get("scores")
            ) for c in conflicts_resolved],
            alignPlus=AlignPlus(Abase=abase, Coverage=coverage, OffSchemaRate=off_schema_rate, Aplus=aplus),
            debug=debug,
            requestId=rid
        )
    except Exception as e:
        log_json("error", rid, "synapse.fill.error", error=str(e))
        return err(500, rid, str(e))


@app.post("/synapse/paths", response_model=PathsResponse)
def synapse_paths(req: PathsRequest = Body(...), request: Request = None):
    """Join path inference with PathScore™."""
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")
    try:
        startE = None; goalE = None; intent_vec = None

        if req.intent:
            intent_vec = encode_intent_to_vocab(req.intent, IEM.vocab)
            metric_op = ((req.intent.get("metric") or {}).get("op"))
            target = req.intent.get("target")
            startE = None
            if metric_op in {"sum","avg","mean","median"}:
                m = match_candidates(IEM, intent_vec, top_k_fields=8, top_k_entities=5, intent_obj=req.intent)
                money_entity = next((sf.entity for sf in m.topFields if sf.roleTop == "money"), None)
                startE = money_entity or (m.topEntities[0].entity if m.topEntities else None)
            if not startE:
                startE = req.intent.get("entity")
            if target:
                goalE = target.split(".")[0] if "." in target else target

        if req.start:
            se, _ = _parse_entity_field(req.start)
            if se: startE = se
        if req.goal:
            ge, _ = _parse_entity_field(req.goal)
            if ge: goalE = ge

        if not startE or not goalE:
            return PathsResponse(ok=True, paths=[], debug={"warn": "start or goal entity missing", "requestId": rid})

        adj = _build_join_graph(IEM)
        raw_paths = _bfs_paths(adj, start=startE, goal=goalE, max_hops=req.maxHops)

        # get active pathScoring config
        from . import synapse
        from .deps import clamp
        ps_cfg = dict(synapse._CONFIG.get("pathScoring", {}))
        warns = []
        
        # merge per-call overrides if present (with clamping)
        if req.pathScoringOverrides:
            for k, v in (req.pathScoringOverrides or {}).items():
                if k in ps_cfg and v is not None:
                    clamped = clamp(f"pathScoring.{k}", float(v))
                    if clamped != float(v):
                        warns.append(f"clamped {k} from {v} to {clamped}")
                    ps_cfg[k] = clamped
        
        # score paths and normalize
        raw = []
        for p in raw_paths:
            s = _path_score(IEM, intent_vec or [0.0]*IEM.dim, p, cfg=ps_cfg)
            edges = [PathEdge(srcEntity=a, srcField=b, dstEntity=c, dstField=d, why=e) for (a,b,c,d,e) in p]
            raw.append((s, edges))
        
        raw.sort(key=lambda t: t[0], reverse=True)
        max_s = raw[0][0] if raw else 1.0
        
        out: List[PathCandidate] = []
        for s, edges in raw[:req.topK]:
            out.append(PathCandidate(path=edges, score=s, hops=len(edges)))

        return PathsResponse(
            ok=True, 
            paths=out, 
            debug={
                "start": startE, 
                "goal": goalE, 
                "maxHops": req.maxHops, 
                "found": len(raw),
                "pathScoringUsed": ps_cfg,
                "scoreMax": max_s,
                "scoreNormHint": "score / scoreMax",
                "warn": warns if warns else None,
                "requestId": rid
            }
        )
    except Exception as e:
        log_json("error", rid, "synapse.paths.error", error=str(e))
        return err(500, rid, str(e))


@app.post("/synapse/explain", response_model=ExplainResponse)
def synapse_explain(req: ExplainRequest = Body(...), request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")
    try:
        # re-use fill (without mutating default behavior) to get pivot + conflicts
        from .models import FillRequest
        fill_req = FillRequest.model_validate({
            "intent": req.intent,
            "topKTargets": req.topKTargets or 6,
            "preferRoles": req.preferRoles or [],
            "fillOverrides": req.fillOverrides or {}
        })
        f = synapse_fill(fill_req, request)
        if isinstance(f, JSONResponse):
            return f
        fobj: FillResponse = f  # type: ignore

        intent_vec = encode_intent_to_vocab(req.intent or {}, IEM.vocab)
        explains: List = []
        for c in fobj.conflicts:
            scored: List = []
            for tgt in c.candidates[:12]:
                ctx = score_target_context(IEM, intent_vec, tgt, intent_obj=req.intent)
                final_score = 0.5*ctx["PathScore"] + 0.35*ctx["CosineContext"] + 0.15*ctx["RoleCoherence"]
                scored.append(CandidateScore(
                    target=tgt,
                    score=float(final_score),
                    PathScore=float(ctx["PathScore"]),
                    CosineContext=float(ctx["CosineContext"]),
                    RoleCoherence=float(ctx["RoleCoherence"]),
                ))
            scored.sort(key=lambda s: s.score, reverse=True)
            explains.append(ExplainItem(slot=c.slot, resolution=c.resolution or scored[0].target, top=scored[:8]))

        resp = ExplainResponse(
            intentFilled=fobj.intentFilled,
            targetChosen=fobj.intentFilled.get("target"),
            explains=explains,
            alignPlus=fobj.alignPlus,
            debug={},
            requestId=rid
        )
        # log
        try:
            from .decision_log import log_decision
            from . import synapse as _syn
            log_decision(
                event="synapse.explain",
                rid=rid,
                payload={
                    "intent": req.intent,
                    "targetChosen": resp.targetChosen,
                    "explains": [e.model_dump() for e in explains]
                },
                path=_syn._CONFIG["logging"]["path"],
                enabled=_syn._CONFIG["logging"]["decisionsEnabled"],
            )
        except Exception as _e:
            log_json("warning", rid, "decision.log.error", error=str(_e))
        return resp
    except Exception as e:
        log_json("error", rid, "synapse.explain.error", error=str(e))
        return err(500, rid, str(e))


@app.get("/examples/coverage")
def get_coverage():
    """Return cached coverage from last /generate call."""
    if COVERAGE_CACHE is None:
        return JSONResponse({"entities": [], "ok": False, "error": "no coverage cached yet. call /generate first."}, status_code=404)
    return {"entities": COVERAGE_CACHE}


@app.post("/generate", response_model=GenerateExampleResponse)
def generate(req: GenerateRequest = Body(...), request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")
    try:
        global COVERAGE_CACHE
        items_raw, cover = generate_all(IEM, per_entity=req.perEntity, strict=req.strict)
        COVERAGE_CACHE = cover  # cache for /examples/coverage endpoint
        written = 0
        cov_path = None
        target_dir = None
        if req.writeFiles:
            # NEW: normalize/guard outDir under OUT_BASE
            target_dir = os.path.abspath(req.outDir)
            if not (target_dir + os.sep).startswith(OUT_BASE + os.sep):
                return err(400, rid, f"outDir must be inside NEURONS_OUT_BASE ({OUT_BASE})")
            written = write_examples(items_raw, target_dir)
            if req.writeCoverage:
                cov_path = write_coverage(cover, target_dir)
        items = [GenIQL(**it) for it in items_raw]
        coverage = [CoverageStat(**c) for c in cover]
        log_json("info", rid, "generate.examples", total=len(items), written=written, outDir=(target_dir if req.writeFiles else None))
        return GenerateExampleResponse(**ok(rid, {"total": len(items), "written": written, "outDir": (target_dir if req.writeFiles else None), "coverageFile": cov_path, "items": items, "coverage": coverage}))
    except Exception as e:
        log_json("error", rid, "generate.error", error=str(e))
        return err(500, rid, str(e))