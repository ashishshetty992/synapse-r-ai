import math, json, logging, re, os
from typing import List, Dict, Any, Optional
from collections import defaultdict, deque, Counter
from math import sqrt

from .models import MatchResponse, ScoredField, ScoredEntity
from .iem import IEMIndex

LOG = logging.getLogger(__name__)

# Default config structure
_DEFAULT_CONFIG = {
    "role_keywords": {},
    "shaping_weights": {"alias_alpha": 0.35, "shape_alpha": 0.12, "shape_beta": 0.10, "metric_alpha": 0.12},
    "entity_blend": {"entityWeight": 0.70, "fieldWeight": 0.30, "entityAliasAlpha": 0.15},
    "centroid_gamma": 0.60,
    "role_alpha_default": 0.12,
    "fill": {
        "unknownPenalty": 0.10,
        "maxUnknownInTopK": 1,
        "preferBonus": 0.08,
        "moneyPivotPenalty": 0.08,
        "osrFromIntentOnly": True,
        "conflict": {
            "pathWeight": 0.50,
            "cosineWeight": 0.35,
            "roleWeight": 0.15,
            "minPathScore": 0.02,
            "lengthPriorBase": 0.92,
            "preferMetricEntityBonus": 0.03,
        },
    },
    "pathScoring": {
        "idPrior": 0.10,
        "fkBonus": 0.05,
        "lengthPriorBase": 0.92,
        "cosineWeight": 0.75,
        "pkToFkBonus": 0.00,
        "fkToPkBonus": 0.00,
        "bridgePenalty": 0.00,
        "hubPenalty": 0.00,
        "hubDegreeThreshold": 8,
        "fanoutPenalty": 0.00,
        "sameEntityTimestampPrior": 0.02,  # NEW: prior for timestamp fields in same entity
    },
    "logging": {
        "decisionsEnabled": True,
        "path": "./logs/decisions.jsonl.gz",
    },
    "alignPlus": {
        "osrZeroWhenNoText": True,
    },
    "salience": {
        "gamma": 0.15,
    },
}

def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

# Try to load merged config from active checkpoint, fallback to defaults
try:
    from .active_cfg import load_active_config_merged
    active_cfg = load_active_config_merged()
    if active_cfg:
        _CONFIG = _deep_merge(_DEFAULT_CONFIG, active_cfg)
    else:
        _CONFIG = dict(_DEFAULT_CONFIG)
except Exception:
    # Fallback to defaults if active config not available
    _CONFIG = dict(_DEFAULT_CONFIG)

def set_runtime_config(cfg: dict):
    """Update _CONFIG with merged active config."""
    global _CONFIG
    if cfg:
        _CONFIG = _deep_merge(_DEFAULT_CONFIG, cfg)
    else:
        _CONFIG = dict(_DEFAULT_CONFIG)

# Salience cache
_SALIENCE_CACHE = None

def _get_salience_map(iem):
    """Get salience map from IEM or compute fallback version."""
    global _SALIENCE_CACHE
    if _SALIENCE_CACHE is not None:
        return _SALIENCE_CACHE
    
    # If IEM already carries salience, use it; else compute a simple version
    sal = {}
    if hasattr(iem, "_salience_map") and isinstance(iem._salience_map, dict):
        sal = iem._salience_map
    else:
        # fallback: PK/FK/UNIQUE/INDEX → weight; name tokens → tiny bump
        base = {}
        for f in iem.fields:
            key = f"{f.entity}.{f.name}"
            w = 0.0
            if getattr(f, "is_pk", False):
                w += 1.0
            if getattr(f, "is_fk", False):
                w += 0.6
            if getattr(f, "is_unique", False):
                w += 0.3
            if getattr(f, "is_indexed", False):
                w += 0.2
            nm = (f.name or "").lower()
            for tok in ("price", "amount", "total", "status", "method", "channel", "city", "country", "segment", "category", "id"):
                if tok in nm:
                    w += 0.1
            base[key] = w
        # normalize 0..1
        mx = max(base.values()) if base else 1.0
        sal = {k: (v / mx) for k, v in base.items()}
    
    _SALIENCE_CACHE = sal
    return sal

def set_role_config(keywords: dict[str, list[str]] | None = None):
    if keywords: 
        _CONFIG["role_keywords"] = {k: list(v) for k, v in keywords.items()}
        print(f"DEBUG: set_role_config called with {len(keywords)} roles")

def set_shaping_weights(weights: dict[str, float] | None = None):
    if weights: 
        _CONFIG["shaping_weights"].update({k: float(v) for k, v in weights.items()})
        print(f"DEBUG: set_shaping_weights called with {len(weights)} weights")

def set_entity_blend(entity_weight: float = None, field_weight: float = None, entity_alias_alpha: float = None, **kwargs):
    if entity_weight is None and "entityWeight" in kwargs:
        entity_weight = float(kwargs["entityWeight"])
    if field_weight is None and "fieldWeight" in kwargs:
        field_weight = float(kwargs["fieldWeight"])
    if entity_alias_alpha is None and "entityAliasAlpha" in kwargs:
        entity_alias_alpha = float(kwargs["entityAliasAlpha"])

    _CONFIG["entity_blend"] = {
        "entityWeight": float(entity_weight if entity_weight is not None else 0.70),
        "fieldWeight": float(field_weight if field_weight is not None else 0.30),
        "entityAliasAlpha": float(entity_alias_alpha if entity_alias_alpha is not None else 0.15),
    }
    print(f"DEBUG: set_entity_blend called with entity_weight={_CONFIG['entity_blend']['entityWeight']}, "
          f"field_weight={_CONFIG['entity_blend']['fieldWeight']}, "
          f"entity_alias_alpha={_CONFIG['entity_blend']['entityAliasAlpha']}")

def set_centroid_gamma(gamma: float):
    _CONFIG["centroid_gamma"] = float(gamma)
    print(f"DEBUG: set_centroid_gamma called with gamma={gamma}")

def set_role_alpha_default(v: float):
    _CONFIG["role_alpha_default"] = float(v)
    print(f"DEBUG: set_role_alpha_default called with v={v}")

def set_fill_config(d: dict | None):
    """Update fill-time knobs (unknownPenalty/maxUnknownInTopK/preferBonus/moneyPivotPenalty/osrFromIntentOnly/conflict)."""
    if not d: return
    cur = _CONFIG["fill"]
    for k in ("unknownPenalty","maxUnknownInTopK","preferBonus","moneyPivotPenalty","osrFromIntentOnly","conflict"):
        if k in d and d[k] is not None:
            if k == "conflict" and isinstance(d[k], dict):
                cur.setdefault("conflict", {}).update(d[k])
            else:
                cur[k] = d[k]
    print(f"DEBUG: set_fill_config -> {cur}")

def set_path_scoring(d: dict | None):
    if not d: return
    ps_cfg = _CONFIG.get("pathScoring", {})
    _CONFIG["pathScoring"] = {
        "idPrior": float(d.get("idPrior", 0.10)),
        "fkBonus": float(d.get("fkBonus", 0.05)),
        "lengthPriorBase": float(d.get("lengthPriorBase", 0.92)),
        "cosineWeight": float(d.get("cosineWeight", 0.75)),
        "pkToFkBonus": float(d.get("pkToFkBonus", ps_cfg.get("pkToFkBonus", 0.00))),
        "fkToPkBonus": float(d.get("fkToPkBonus", ps_cfg.get("fkToPkBonus", 0.00))),
        "bridgePenalty": float(d.get("bridgePenalty", ps_cfg.get("bridgePenalty", 0.00))),
        "hubPenalty": float(d.get("hubPenalty", ps_cfg.get("hubPenalty", 0.00))),
        "hubDegreeThreshold": int(d.get("hubDegreeThreshold", ps_cfg.get("hubDegreeThreshold", 8))),
        "fanoutPenalty": float(d.get("fanoutPenalty", ps_cfg.get("fanoutPenalty", 0.00))),
        # Safety clamp: sameEntityTimestampPrior between 0.0 and 0.5
        "sameEntityTimestampPrior": max(0.0, min(0.5, float(d.get("sameEntityTimestampPrior", ps_cfg.get("sameEntityTimestampPrior", 0.02))))),
    }
    print(f"DEBUG: set_path_scoring -> {_CONFIG['pathScoring']}")
    LOG.info(f"set_path_scoring applied: {_CONFIG['pathScoring']}")

def set_logging_cfg(d: dict | None):
    if not d: return
    cur = _CONFIG["logging"]
    if "decisionsEnabled" in d: cur["decisionsEnabled"] = bool(d["decisionsEnabled"])
    if "path" in d and d["path"]: cur["path"] = str(d["path"])

def set_alignplus_cfg(d: dict | None):
    if not d: 
        return
    cur = _CONFIG.get("alignPlus", {})
    if "osrZeroWhenNoText" in d and d["osrZeroWhenNoText"] is not None:
        cur["osrZeroWhenNoText"] = bool(d["osrZeroWhenNoText"])
    _CONFIG["alignPlus"] = cur
    print(f"DEBUG: set_alignplus_cfg -> {_CONFIG['alignPlus']}")

# def set_path_scoring_config(d: dict | None):
#     """
#     DEPRECATED: Use set_path_scoring() instead.
#     Update PathScore™ knobs from shaping.json["pathScoring"].
#     Keys: idPrior, fkBonus, lengthPriorBase, cosineWeight
#     """
#     if not d: return
#     cur = _CONFIG["pathScoring"]
#     for k in ("idPrior", "fkBonus", "lengthPriorBase", "cosineWeight"):
#         if k in d and d[k] is not None:
#             cur[k] = float(d[k])
#     print(f"DEBUG: set_path_scoring_config -> {cur}")


def _bag_from_intent(d: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(str(d.get("ask","")))
    for k in ["target","targets","metric","filters","timeWindow","orderBy","comparePeriods","select"]:
        v = d.get(k)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            parts.append(json.dumps(v, separators=(",",":")))
        elif isinstance(v, list):
            for it in v:
                parts.append(str(it))
    return " ".join(parts).lower()


def _role_prior_from_intent(d: Dict[str, Any]) -> Dict[str, float]:
    text = _bag_from_intent(d)
    print(f"DEBUG: _role_prior_from_intent - text='{text}'")
    print(f"DEBUG: _role_prior_from_intent - role_keywords={_CONFIG.get('role_keywords', {})}")
    
    pri = defaultdict(float)
    for role, kws in (_CONFIG["role_keywords"] or {}).items():
        for w in kws:
            if w in text:
                pri[role] += 1.0
                print(f"DEBUG: Found keyword '{w}' for role '{role}'")
    
    s = sum(pri.values()) or 1.0
    result = {k: v/s for k, v in pri.items()}
    print(f"DEBUG: _role_prior_from_intent result: {result}")
    return result


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _role_prior_from_centroids(intent_vec: List[float], role_centroids: Dict[str, List[float]]) -> Dict[str, float]:
    scores = {}
    for role, cvec in role_centroids.items():
        scores[role] = _cosine(intent_vec, cvec)
    scores = {k: max(0.0, v) for k, v in scores.items()}
    s = sum(scores.values()) or 1.0
    return {k: v/s for k, v in scores.items()}


def _blend_priors(keyword_prior: Dict[str, float],
                  centroid_prior: Dict[str, float],
                  gamma: float = 0.6) -> Dict[str, float]:
    """
    gamma=0 → only keywords, gamma=1 → only centroids
    """
    roles = set(list(keyword_prior.keys()) + list(centroid_prior.keys()))
    out = {}
    for r in roles:
        kw = keyword_prior.get(r, 0.0)
        ce = centroid_prior.get(r, 0.0)
        out[r] = (1.0 - gamma) * kw + gamma * ce
    s = sum(out.values()) or 1.0
    return {k: v/s for k, v in out.items()}


def _guess_intent_role_tokens(intent_text: str) -> Dict[str, float]:
    """
    Very light role prior from keywords inside the serialized intent text.
    """
    text = intent_text.lower()
    scores = defaultdict(float)
    tokens = {
        "money": ["revenue", "amount", "price", "total", "gmv", "sales"],
        "timestamp": ["day", "date", "week", "month", "year", "created_at"],
        "geo": ["city", "country", "state", "region"],
        "category": ["segment", "category", "status", "method", "channel"],
        "quantity": ["count", "qty", "units", "orders"],
        "text": ["name", "title"],
        "id": ["id", "order_id", "customer_id", "product_id"],
    }
    for r, kws in tokens.items():
        for k in kws:
            if k in text:
                scores[r] += 1.0
    s = sum(scores.values()) or 1.0
    return {k: v / s for k, v in scores.items()}


def match_candidates(
    iem: IEMIndex,
    intent_vec: List[float],
    top_k_fields: int = 8,
    top_k_entities: int = 5,
    role_alpha: Optional[float] = None,
    intent_obj: Dict[str, Any] | None = None,
    alias_alpha: Optional[float] = None,
    shape_alpha: Optional[float] = None,
    shape_beta: Optional[float] = None,
    metric_alpha: Optional[float] = None,
) -> MatchResponse:
    """
    Scores (cosine) each field vector against the intent vector,
    with small role/ask/alias nudges.
    """
    if role_alpha is None:
        role_alpha = _CONFIG["role_alpha_default"]

    print(f"DEBUG: match_candidates - role_alpha={role_alpha}, _CONFIG keys={list(_CONFIG.keys())}")
    print(f"DEBUG: _CONFIG role_keywords={len(_CONFIG.get('role_keywords', {}))}")
    print(f"DEBUG: _CONFIG shaping_weights={_CONFIG.get('shaping_weights', {})}")
    print(f"DEBUG: _CONFIG entity_blend={_CONFIG.get('entity_blend', {})}")
    print(f"DEBUG: _CONFIG centroid_gamma={_CONFIG.get('centroid_gamma', 'NOT_SET')}")

    if intent_obj:
        keyword_prior = _role_prior_from_intent(intent_obj)
        print(f"DEBUG: keyword_prior from intent_obj: {keyword_prior}")
    else:
        intent_signature = " ".join([str(round(x, 3)) for x in intent_vec[:32]])
        keyword_prior = _guess_intent_role_tokens(intent_signature)
        print(f"DEBUG: keyword_prior from signature: {keyword_prior}")

    centroid_prior: Dict[str, float] = {}
    if getattr(iem, "role_centroids", None):
        print(f"DEBUG: role_centroids available: {list(iem.role_centroids.keys())}")
        sims = {}
        a = intent_vec
        for r, cvec in iem.role_centroids.items():
            b = cvec
            dot = sum(x*y for x, y in zip(a, b))
            sims[r] = max(0.0, float(dot))
        s = sum(sims.values()) or 1.0
        centroid_prior = {k: v/s for k, v in sims.items()}
        print(f"DEBUG: centroid_prior computed: {centroid_prior}")
    else:
        print("DEBUG: No role_centroids available in IEM")

    gamma = _CONFIG["centroid_gamma"]
    role_prior = _blend_priors(keyword_prior, centroid_prior, gamma=gamma)
    print(f"DEBUG: role_prior blended with gamma={gamma}: {role_prior}")

    def _parse_targets(t):
        if not t:
            return []
        if isinstance(t, str):
            parts = t.split(".")
            return [(parts[0], parts[1])] if len(parts) == 2 else []
        if isinstance(t, list):
            out = []
            for x in t:
                if isinstance(x, str) and "." in x:
                    p = x.split(".")
                    out.append((p[0], p[1]))
            return out
        return []

    target = (intent_obj or {}).get("target") or (intent_obj or {}).get("targets")
    target_pairs = _parse_targets(target)

    scored_fields: List[ScoredField] = []
    for f in iem.fields:
        base = _cosine(intent_vec, f.vec)

        top_role = max(f.role.items(), key=lambda kv: kv[1])[0] if f.role else None
        role_bonus = role_prior.get(top_role or "", 0.0) * role_alpha

        ask = (intent_obj or {}).get("ask")
        metric = (intent_obj or {}).get("metric", {})
        metric_op = (metric or {}).get("op")

        norm = lambda s: (s or "").lower().strip()
        def variants(s: str) -> set[str]:
            s = norm(s)
            return {s, s.replace("_", " "), s.replace(" ", "_")}

        target_field_tokens: set[str] = set()
        for (_, b) in target_pairs:
            target_field_tokens |= variants(b)

        is_target_exact = (f.entity, f.name) in target_pairs
        aliases = (f.aliases or []) + [f.name]
        has_alias_hit = any(
            (norm(al) in target_field_tokens) or (variants(al) & target_field_tokens)
            for al in aliases
        )

        sw = _CONFIG["shaping_weights"]
        alias_alpha_val  = alias_alpha if alias_alpha is not None else sw["alias_alpha"]
        shape_alpha_val   = shape_alpha if shape_alpha is not None else sw["shape_alpha"]
        shape_beta_val    = shape_beta if shape_beta is not None else sw["shape_beta"]
        metric_alpha_val  = metric_alpha if metric_alpha is not None else sw["metric_alpha"]

        alias_bonus = alias_alpha_val if (is_target_exact or has_alias_hit) else 0.0

        role_shape = 0.0
        if ask == "top_k" and metric_op in {"count", "distinct_count"}:
            if top_role in {"geo", "category", "text"}:
                role_shape += shape_alpha_val
            if top_role in {"money"}:
                role_shape -= shape_beta_val

        if metric_op in {"sum", "avg", "mean", "median"} and top_role == "money":
            role_shape += metric_alpha_val

        score = base * (1.0 + role_bonus) + alias_bonus + role_shape
        
        # Apply salience multiplier
        SALIENCE = getattr(iem, "_salience_map", None)
        gamma = float((_CONFIG.get("salience") or {}).get("gamma", 0.15)) if "salience" in _CONFIG else 0.15
        def _apply_salience(target, score):
            if not SALIENCE:
                return score
            s = float(SALIENCE.get(target, 0.0))
            return float(score) * (1.0 + gamma * s)
        
        target_key = f"{f.entity}.{f.name}"
        score = _apply_salience(target_key, score)
        
        scored_fields.append(ScoredField(entity=f.entity, name=f.name, score=score, roleTop=top_role))

    scored_fields.sort(key=lambda x: x.score, reverse=True)
    top_fields = scored_fields[:top_k_fields]

    eb = _CONFIG["entity_blend"]

    top_entities: List[ScoredEntity] = []
    if getattr(iem, "entities", None):
        per_ent_fields = defaultdict(list)
        for sf in scored_fields[: max(64, top_k_fields * 4)]:
            per_ent_fields[sf.entity].append(sf.score)

        target_entities = {a for (a, _) in target_pairs}
        e_w, f_w, e_alias = eb["entityWeight"], eb["fieldWeight"], eb["entityAliasAlpha"]

        es = []
        for e in iem.entities:
            cos = _cosine(intent_vec, e.vec)
            field_avg = sum(per_ent_fields[e.name]) / max(1, len(per_ent_fields[e.name]))
            alias_bonus = e_alias if e.name in target_entities else 0.0
            score = e_w * float(cos) + f_w * float(field_avg) + alias_bonus
            es.append(ScoredEntity(entity=e.name, score=score))
        es.sort(key=lambda x: x.score, reverse=True)
        top_entities = es[:top_k_entities]
    else:
        per_ent = defaultdict(list)
        for sf in scored_fields[: max(64, top_k_fields * 4)]:
            per_ent[sf.entity].append(sf.score)
        ent_scores = [ScoredEntity(entity=e, score=(sum(vs)/len(vs) if vs else 0.0)) for e,vs in per_ent.items()]
        ent_scores.sort(key=lambda x: x.score, reverse=True)
        top_entities = ent_scores[:top_k_entities]

    return MatchResponse(
        ok=True,
        vocabDim=iem.dim,
        topFields=top_fields,
        topEntities=top_entities,
        debug={
            "rolePrior": role_prior,
            "keywordPrior": keyword_prior,
            "centroidPrior": centroid_prior,
            "blendWeights": {"keyword": round(1-gamma,2), "centroid": round(gamma,2)},
            "entityBlend": eb,
            "roleAlphaUsed": role_alpha,
            "shapingWeights": {
                "aliasAlpha": alias_alpha_val,
                "shapeAlpha": shape_alpha_val,
                "shapeBeta": shape_beta_val,
                "metricAlpha": metric_alpha_val,
            },
            "consideredFields": len(iem.fields),
            "vocabDim": iem.dim,
            "ask": (intent_obj or {}).get("ask"),
            "metricOp": ((intent_obj or {}).get("metric") or {}).get("op"),
            "targetPairs": target_pairs,
        },
    )



def _parse_entity_field(s: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not s: return None, None
    parts = s.split(".")
    if len(parts) == 2: return parts[0], parts[1]
    return s, None


def _entity_of_field_name(field_full: str) -> Optional[str]:
    if "." in field_full:
        return field_full.split(".")[0]
    return None


def _coverage_score(intent: Dict[str, Any]) -> float:
    slots = 0; hit = 0
    for key in ["ask", "metric", "target"]:
        slots += 1
        if key in intent and intent[key]:
            hit += 1
    if "timeWindow" in intent: slots += 1; hit += 1
    return hit / max(1, slots)


def _off_schema_rate(tokens: list[str], mapped: set[str]) -> float:
    if not tokens: return 1.0
    bad = sum(1 for t in tokens if t not in mapped)
    return bad / len(tokens)


def compute_alignplus(Abase: float, intent: Dict[str, Any],
                      tokens: list[str], mapped_terms: set[str],
                      shaping_cfg: dict = None) -> dict:
    """
    Compute AlignPlus with deterministic clamps.
    Prevents scores >1.0 due to salience/bonus inflation.
    """
    cov = _coverage_score(intent)

    # merge: runtime _CONFIG["alignPlus"] is the source of truth; shaping_cfg (if provided)
    # can still override for backward-compat.
    align_cfg = dict(_CONFIG.get("alignPlus", {}))
    if shaping_cfg:
        align_cfg.update(shaping_cfg.get("alignPlus", {}))
    osr_zero_when_no_text = bool(align_cfg.get("osrZeroWhenNoText", True))
    if not tokens and osr_zero_when_no_text:
        osr = 0.0
    else:
        osr = _off_schema_rate(tokens, mapped_terms)

    # Clamp Abase
    Abase = max(0.0, min(1.0, float(Abase)))
    
    # Clamp coverage
    coverage = max(0.0, min(1.0, float(cov)))
    
    # Clamp OSR
    osr = max(0.0, min(1.0, float(osr)))

    # Compute Aplus
    Aplus = Abase * (1.0 - osr) * (0.5 + 0.5 * coverage)
    
    # Final clamp (just in case)
    Aplus = max(0.0, min(1.0, Aplus))
    
    return {
        "Abase": Abase,
        "Coverage": coverage,
        "OffSchemaRate": osr,
        "Aplus": Aplus,
    }


def _entity_graph(iem: IEMIndex):
    """Build adjacency by entity via FK-like edges (heuristic: *_id)."""
    adj = {e: set() for e in {f.entity for f in iem.fields}}
    by_entity = {}
    for f in iem.fields:
        by_entity.setdefault(f.entity, []).append(f)
    for e, fs in by_entity.items():
        for f in fs:
            if f.name.endswith("_id"):
                tgt = f.name[:-3]  # 'order_id' -> 'order'
                if tgt in by_entity:
                    adj[e].add(tgt)
                    adj[tgt].add(e)
    return adj

def _centrality(adj):
    """Degree centrality (cheap & deterministic)."""
    return {e: float(len(nei)) for e, nei in adj.items()}

def _constraint_weight(field):
    """Deterministic hints from name; extend later for real metadata."""
    w = 0.0
    if field.name in ("id", "order_id", "customer_id", "product_id"):
        w += 0.5
    if field.name.endswith("_id"):
        w += 0.3
    if field.name in ("created_at",):
        w += 0.2
    return w

def _name_confidence(glossary_terms, field):
    """Check how many glossary terms appear in field name."""
    toks = re.findall(r"[a-z0-9]+", f"{field.entity}.{field.name}".lower())
    s = len(set(toks) & set(glossary_terms))
    return min(1.0, 0.2 * s)

def compute_salience(iem: IEMIndex, glossary: Dict[str, Any]):
    """
    Compute salience score for each field based on:
    - Entity centrality (45%)
    - Constraint weight (35%)
    - Name confidence from glossary (20%)
    Returns normalized salience map (0..1).
    """
    adj = _entity_graph(iem)
    cent = _centrality(adj)
    terms = set((glossary or {}).get("business_terms", []))
    sal = {}
    for f in iem.fields:
        s = 0.45 * cent.get(f.entity, 0.0)
        s += 0.35 * _constraint_weight(f)
        s += 0.20 * _name_confidence(terms, f)
        sal[f"{f.entity}.{f.name}"] = float(s)
    # normalize 0..1
    if sal:
        mx = max(sal.values()) or 1.0
        for k in sal:
            sal[k] = sal[k] / mx
    return sal

def default_role_prior_vector(iem: IEMIndex):
    """
    Derive role prior vector from role composition (share) + centrality bias.
    Returns dict with role probabilities.
    """
    adj = _entity_graph(iem)
    cent = _centrality(adj)
    role_sum = Counter()
    for f in iem.fields:
        rt = (max((f.role or {}).items(), key=lambda kv: kv[1])[0] if f.role else "unknown")
        role_sum[rt] += 1 + 0.15 * cent.get(f.entity, 0.0)
    total = sum(role_sum.values()) or 1.0
    def pct(x):
        return float(role_sum.get(x, 0.0)) / total
    return {
        "id": pct("id"),
        "timestamp": pct("timestamp"),
        "money": pct("money"),
        "geo": pct("geo"),
        "category": pct("category"),
        "quantity": pct("quantity"),
        "text": pct("text"),
        "unknown": pct("unknown")
    }

def _build_join_graph(iem: IEMIndex) -> dict[str, list[tuple[str,str,str,str]]]:
    """Return adjacency: entity -> list of (srcField, dstEntity, dstField, why)"""
    adj: dict[str, list[tuple[str,str,str,str]]] = defaultdict(list)
    for j in (getattr(iem, "joins", []) or []):
        # keep direction so we can differentiate pk→fk vs fk→pk later
        adj[j.srcEntity].append((j.srcField, j.dstEntity, j.dstField, "fk"))
        adj[j.dstEntity].append((j.dstField, j.srcEntity, j.srcField, "fk_rev"))
    return adj

# --- tiny perf caches (safe, bounded) ---
_BFS_CACHE: dict[tuple[int,str,str,int], list[list[tuple[str,str,str,str]]]] = {}
_COS_CACHE: dict[tuple[int, str, str], float] = {}


def _bfs_paths(adj: dict[str, list[tuple[str,str,str,str]]],
               start: str, goal: str, max_hops: int = 2) -> list[list[tuple[str,str,str,str]]]:
    """BFS over entities; edges carry (srcField, dstEntity, dstField, why)"""
    if start == goal: return [[]]
    # cache by (graph_id, start, goal, max_hops). graph_id≈|adj| to avoid keeping iem.
    cache_key = (len(adj), start, goal, max_hops)
    if cache_key in _BFS_CACHE:
        return _BFS_CACHE[cache_key]
    paths = []
    queue = deque()
    queue.append((start, [], set([start])))
    while queue:
        curr_ent, path, seen = queue.popleft()
        if len(path) > max_hops: 
            continue
        for (srcF, dstE, dstF, why) in adj.get(curr_ent, []):
            if dstE in seen: 
                continue
            new_path = path + [(curr_ent, srcF, dstE, dstF, why)]
            if dstE == goal:
                paths.append(new_path)
            else:
                queue.append((dstE, new_path, seen | set([dstE])))
    _BFS_CACHE[cache_key] = paths
    return paths


def _top_role_of_field(f) -> str:
    if not getattr(f, "role", None): return "unknown"
    return max(f.role.items(), key=lambda kv: kv[1])[0]

def _field_map(iem) -> dict:
    return {(f.entity, f.name): f for f in iem.fields}

def _cos(a: list[float], b: list[float]) -> float:
    return sum(x*y for x, y in zip(a, b)) if a and b else 0.0

def _debug_path_scoring_source():
    """Helper to build pathScoring debug dict with all supported keys."""
    cfg = _CONFIG.get("pathScoring", {})
    # Include *all* supported keys, even if 0.0, so the UI/curl shows them
    keys = [
        "idPrior", "fkBonus", "lengthPriorBase", "cosineWeight",
        "pkToFkBonus", "fkToPkBonus", "bridgePenalty", "hubPenalty",
        "hubDegreeThreshold", "fanoutPenalty",
        # NEW
        "sameEntityTimestampPrior",
    ]
    out = {k: float(cfg.get(k, 0.0)) for k in keys}
    # Get experiment from module-level variable if set
    experiment = globals().get("_current_experiment", "")
    out["_source"] = "tenant" if experiment == "baseline-only" else "active"
    return out

def _role_prob(f, role: str) -> float:
    if not getattr(f, "role", None): return 0.0
    return float(f.role.get(role, 0.0))

def _metric_entity_from_intent(intent: dict[str, Any]) -> str | None:
    tgt = (intent or {}).get("target")
    if isinstance(tgt, str) and "." in tgt:
        return tgt.split(".")[0]
    return None

def score_target_context(iem, intent_vec, target_full: str, intent_obj: dict | None = None) -> dict:
    """
    Join-aware scoring for a candidate target relative to the current intent:
      - CosineContext: cosine(intent, field)
      - RoleCoherence: top-role prob on the field
      - PathScore: best FK path prior from a pivot entity inferred from intent
                   (money entity for sum/avg; else entity of explicit target if present).
    """
    try:
        entity, field = target_full.split(".", 1)
    except Exception:
        return {"CosineContext": 0.0, "RoleCoherence": 0.0, "PathScore": 0.0}

    fmap = {(f.entity, f.name): f for f in iem.fields}
    f = fmap.get((entity, field))
    if not f:
        return {"CosineContext": 0.0, "RoleCoherence": 0.0, "PathScore": 0.0}

    cos = max(0.0, sum(x*y for x, y in zip(intent_vec, f.vec)))
    role_top = max(f.role.items(), key=lambda kv: kv[1])[0] if f.role else "unknown"
    role_prob = float((f.role or {}).get(role_top, 0.0))

    pivot_entity = None
    if intent_obj:
        metric_op = ((intent_obj.get("metric") or {}).get("op") or "").lower()
        if metric_op in {"sum", "avg", "mean", "median"}:
            best_e, best_s = None, -1.0
            for ff in iem.fields:
                if "money" in (ff.role or {}):
                    s = sum(x*y for x, y in zip(intent_vec, ff.vec))
                    if s > best_s:
                        best_s, best_e = s, ff.entity
            pivot_entity = best_e
        if not pivot_entity:
            t = intent_obj.get("target")
            if isinstance(t, str) and "." in t:
                pivot_entity = t.split(".", 1)[0]

    if not pivot_entity:
        pivot_entity = entity

    if pivot_entity == entity:
        ps_cfg = _CONFIG["pathScoring"]
        id_prior = float(ps_cfg.get("idPrior", 0.0))
        ts_prior = float(ps_cfg.get("sameEntityTimestampPrior", 0.0))
        prior = 0.0

        debug_meta = {
            "timestampPriorApplied": False,
            "timestampPriorValue": 0.0,
            "localityDecayApplied": False
        }

        # timestamp same-entity prior
        if (f.role or {}).get("timestamp", 0.0) > 0.0:
            prior = ts_prior
            debug_meta["timestampPriorApplied"] = True
            debug_meta["timestampPriorValue"] = ts_prior
        # preserve the existing id prior behavior for *_id fields
        if field.endswith("_id"):
            prior += id_prior

        out = {
            "CosineContext": float(cos),
            "RoleCoherence": role_prob,
            "PathScore": float(prior),
            "_debug": debug_meta
        }
        return out

    adj = _build_join_graph(iem)
    raw_paths = _bfs_paths(adj, start=pivot_entity, goal=entity, max_hops=3)

    if not raw_paths:
        return {"CosineContext": float(cos), "RoleCoherence": role_prob, "PathScore": 0.0}

    ps_cfg = _CONFIG["pathScoring"]
    id_prior = float(ps_cfg.get("idPrior", 0.1))
    fk_bonus = float(ps_cfg.get("fkBonus", 0.05))
    length_decay = float(ps_cfg.get("lengthPriorBase", 0.92))
    cos_w = float(ps_cfg.get("cosineWeight", 0.75))
    
    # NEW knobs picked from merged shaping/principles
    pk_to_fk = float(ps_cfg.get("pkToFkBonus", 0.0))
    fk_to_pk = float(ps_cfg.get("fkToPkBonus", 0.0))
    bridge_pen = float(ps_cfg.get("bridgePenalty", 0.0))
    hub_pen = float(ps_cfg.get("hubPenalty", 0.0))
    hub_thr = int(ps_cfg.get("hubDegreeThreshold", 8))
    fanout_pen = float(ps_cfg.get("fanoutPenalty", 0.0))
    
    # Pre-compute degrees once
    adj = _build_join_graph(iem)
    deg = defaultdict(int)
    for u, edges in adj.items():
        deg[u] = len(edges)

    best = 0.0
    name_map = {(ff.entity, ff.name): ff for ff in iem.fields}

    for edges in raw_paths:
        hop_priors = []
        hop_cos = []
        for (srcE, srcF, dstE, dstF, why) in edges:
            prior = 0.0
            # base priors
            if srcF.endswith("_id") or dstF.endswith("_id"):
                prior += id_prior
            if "fk" in (why or ""):  # your existing "fk" labeling
                prior += fk_bonus
            
            # NEW: direction bonuses if your "why" encodes direction
            wy = (why or "").lower()
            if "pk->fk" in wy or wy == "fk":  # forward PK->FK
                prior += pk_to_fk
            elif "fk->pk" in wy or wy == "fk_rev":  # reverse FK->PK
                prior += fk_to_pk
            
            # NEW: structural penalties
            if deg.get(srcE, 0) >= hub_thr or deg.get(dstE, 0) >= hub_thr:
                prior -= hub_pen
            if "bridge" in wy:
                prior -= bridge_pen
            
            # optional: if an edge fans out a lot (heuristic)
            if deg.get(srcE, 0) > deg.get(dstE, 0) + 4:
                prior -= fanout_pen

            # cosine on hop endpoints (existing)
            a = name_map.get((srcE, srcF))
            b = name_map.get((dstE, dstF))
            if a and b:
                cval = 0.5 * max(0.0, sum(x*y for x, y in zip(intent_vec, a.vec))) \
                     + 0.5 * max(0.0, sum(x*y for x, y in zip(intent_vec, b.vec)))
            else:
                cval = 0.0

            hop_priors.append(prior)
            hop_cos.append(cval)

        path_prior = sum(hop_priors) / max(1, len(hop_priors))
        path_cos = sum(hop_cos) / max(1, len(hop_cos))
        combined = (cos_w * path_cos) + ((1.0 - cos_w) * path_prior)
        combined *= (length_decay ** (len(edges) - 1))
        best = max(best, combined)

    # If 1-hop timestamp found, apply locality decay flag for transparency
    debug_meta = {
        "timestampPriorApplied": False,
        "timestampPriorValue": 0.0,
        "localityDecayApplied": False
    }
    if raw_paths and len(raw_paths[0]) == 1 and (f.role or {}).get("timestamp", 0.0) > 0.0:
        ts_prior = float(ps_cfg.get("sameEntityTimestampPrior", 0.0))
        best = max(best, 0.4 * ts_prior)
        debug_meta["timestampPriorApplied"] = True
        debug_meta["timestampPriorValue"] = 0.4 * ts_prior
        debug_meta["localityDecayApplied"] = True
        return {
            "CosineContext": float(cos),
            "RoleCoherence": role_prob,
            "PathScore": float(best),
            "_debug": debug_meta
        }

    return {"CosineContext": float(cos), "RoleCoherence": role_prob, "PathScore": float(best)}

def _is_generic_category(term: str | None) -> bool:
    """Check if a term is a generic category word."""
    if not term:
        return False
    t = term.lower()
    return t in {"category", "group", "class", "type", "segment"}

def _package_conflict(slot_name: str, scored: list, chosen_idx: int, why: dict | None = None, rule_override: bool = False):
    """
    Build the conflict dict block with per-candidate details. `scored` is a list of dicts containing:
    - key, score, finalScore, reason, PathScore, CosineContext, RoleCoherence, LocalBonus, weights, (optional) timestampInfo
    `chosen_idx` is the index in `scored` that we select (may be a rule override).
    """
    resolution = scored[chosen_idx]["key"] if scored and chosen_idx < len(scored) else None
    # Compute transparency fields
    best_numeric = max((c.get("finalScore", c.get("score", 0.0)) for c in scored), default=0.0)
    chosen_num = scored[chosen_idx].get("finalScore", scored[chosen_idx].get("score", 0.0)) if scored and chosen_idx < len(scored) else 0.0
    # If a specific rule was applied (rule_override=True), we mark the decision as ruleOverride, else finalScore.
    # The presence of preferSameEntityTimestampDelta in why just indicates the threshold is available, not that it was applied.
    selected_by = "ruleOverride" if rule_override else "finalScore"
    return {
        "slot": slot_name,
        "candidates": [c["key"] for c in scored],
        "resolution": resolution,
        "selectedBy": selected_by,
        "decisionScore": float(chosen_num),
        "bestNumericScore": float(best_numeric),
        "why": why or {},
        "scores": [{"target": s["key"], "score": s["score"], "finalScore": s.get("finalScore", s["score"]), "reason": s.get("reason", "normal"), **s["why"]} for s in scored]
    }

def _unknown_backoff(slot_name: str, candidates: list, ctx: dict):
    """
    Cascade fallback for unknown/ambiguous slot resolution:
    1) Honor roleTop match if confident (score >= 0.80)
    2) Emergent tags from principles.json
    3) Salience within same entity as pivot
    4) Final: best PathScore
    """
    if not candidates:
        return None
    
    # 1) honor roleTop match if confident
    confident = [c for c in candidates if getattr(c, "roleTop", "") == slot_name and getattr(c, "score", 0.0) >= 0.80]
    if confident:
        return confident[0]
    
    # 2) emergent tags (status-like) -> we'll plug later through principles.json
    emergent = ctx.get("emergent", {}).get(slot_name, [])
    for t in emergent:
        for c in candidates:
            if f"{c.entity}.{c.name}" == t:
                return c
    
    # 3) salience within same entity as pivot
    pivot_entity = None
    pivot_obj = ctx.get("pivot")
    if pivot_obj:
        if hasattr(pivot_obj, "entity"):
            pivot_entity = pivot_obj.entity
        elif isinstance(pivot_obj, dict):
            pivot_entity = pivot_obj.get("entity")
    
    if pivot_entity:
        same = [c for c in candidates if c.entity == pivot_entity]
        if same:
            salience_map = ctx.get("salience", {})
            same = sorted(same, key=lambda x: salience_map.get(f"{x.entity}.{x.name}", 0.0), reverse=True)
            return same[0]
    
    # 4) final: best PathScore
    path_scores = ctx.get("pathScores", {})
    return sorted(candidates, key=lambda x: path_scores.get(f"{x.entity}.{x.name}", 0.0), reverse=True)[0]

def resolve_conflicts(
    iem,
    intent_vec: list[float],
    intent_obj: dict[str, Any],
    raw_fields: list,
    pivot: Any | None,
    match_debug: dict,
) -> tuple[list, str | None]:
    """
    Returns (conflicts[], maybe_new_target_str).
    Each conflict note: {slot, candidates:[...], resolution, why:{...}}
    """
    fill_cfg = _CONFIG.get("fill", {})
    unknown_backoff = bool(fill_cfg.get("unknownBackoff", False))
    ask = (intent_obj.get("ask") or "").lower()
    
    cf = fill_cfg.get("conflict", {})
    w_path = float(cf.get("pathWeight", 0.5))
    w_cos  = float(cf.get("cosineWeight", 0.35))
    w_role = float(cf.get("roleWeight", 0.15))
    min_path = float(cf.get("minPathScore", 0.02))
    len_base = float(cf.get("lengthPriorBase", 0.92))
    metric_bonus = float(cf.get("preferMetricEntityBonus", 0.03))

    name_map = _field_map(iem)
    adj = _build_join_graph(iem)

    metric_entity = _metric_entity_from_intent(intent_obj) or (match_debug.get("topEntities", [{}])[0:1] or [None])[0]
    if isinstance(metric_entity, dict):
        metric_entity = metric_entity.get("entity")
    money_field = None
    for sf in raw_fields:
        if getattr(sf, "roleTop", None) == "money":
            money_field = name_map.get((sf.entity, sf.name))
            break

    roles_to_check = {"geo", "timestamp", "category"}
    by_role: dict[str, dict[str, Any]] = {r: {} for r in roles_to_check}

    for sf in raw_fields:
        r = getattr(sf, "roleTop", None) or _top_role_of_field(name_map.get((sf.entity, sf.name)))
        if r not in roles_to_check:
            continue
        prev = by_role[r].get(sf.entity)
        if (prev is None) or (sf.score > prev.score):
            by_role[r][sf.entity] = sf

    conflicts = []
    new_target = None

    # Load principles for emergent tags
    try:
        from .config import load_principles
        active_principles = load_principles()
    except Exception:
        active_principles = {"emergent": {}}

    # Get salience map from IEM if available
    salience_map = getattr(iem, "_salience_map", {}) or {}

    for role, ent_map in by_role.items():
        if len(ent_map) < 2:
            continue

        # UnknownBackoff for generic category asks
        slot_name = role
        if slot_name == "category" and unknown_backoff:
            requested = (intent_obj.get("target") or "")
            if _is_generic_category(requested):
                # filter category-like and status-like
                catish = []
                sal_map = _get_salience_map(iem)
                candidates_for_backoff = []
                for ent, sf in ent_map.items():
                    candF = name_map.get((sf.entity, sf.name))
                    if candF:
                        roleTop = getattr(sf, "roleTop", None) or _top_role_of_field(candF)
                        candidates_for_backoff.append((sf, roleTop))
                
                for sf, roleTop in candidates_for_backoff:
                    key = f"{sf.entity}.{sf.name}"
                    if roleTop in {"category"} or "status" in (sf.name or "").lower():
                        sc = float(sf.score)
                        # prefer categorical dims for top_k over "status" (configurable)
                        prefer_cat = float(fill_cfg.get("conflict", {}).get("preferCategoryForTopK", 0.02))
                        pref = prefer_cat if (ask == "top_k" and roleTop == "category") else 0.0
                        sc = sc + pref
                        # salience tie-break
                        gamma = float((_CONFIG.get("salience") or {}).get("gamma", 0.0))
                        sc = sc * (1.0 + gamma * float(sal_map.get(key, 0.0)))
                        catish.append((sc, sf, key))
                
                if catish:
                    catish.sort(key=lambda z: z[0], reverse=True)
                    best_sc, best_sf, best_key = catish[0]
                    # Convert to scored format for _package_conflict (with transparency fields)
                    scored_format = [{"key": k, "score": sc, "finalScore": sc, "reason": "unknownBackoff", "why": {"unknownBackoff": True}} for sc, _, k in catish]
                    why_dict = {
                        "reason": "unknownBackoff",
                        "ask": ask,
                        "salApplied": float((_CONFIG.get("salience") or {}).get("gamma", 0.0))
                    }
                    # Use _package_conflict for consistency and transparency
                    conflict = _package_conflict(slot_name, scored_format, 0, why_dict, rule_override=True)
                    conflicts.append(conflict)
                    if pivot and getattr(pivot, "roleTop", None) == slot_name and f"{pivot.entity}.{pivot.name}" != best_key:
                        new_target = best_key
                    continue  # Skip normal resolution for this slot

        scored = []
        path_scores_local = {}  # local map for pathScores
        for ent, sf in ent_map.items():
            candF = name_map.get((sf.entity, sf.name))
            key = f"{sf.entity}.{sf.name}"
            
            # Use score_target_context to get proper PathScore (includes sameEntityTimestampPrior)
            ctx_score = score_target_context(iem, intent_vec, key, intent_obj)
            path_score = float(ctx_score.get("PathScore", 0.0))
            # Extract debug metadata if present
            debug_meta = ctx_score.get("_debug", {})
            
            # If no path score from context and metric_entity differs, try path-based scoring
            if path_score == 0.0 and metric_entity and metric_entity != sf.entity:
                raw_paths = _bfs_paths(adj, start=metric_entity, goal=sf.entity, max_hops=3)
                if raw_paths:
                    base_ps = _path_score(iem, intent_vec, raw_paths[0])
                    hops = len(raw_paths[0])
                    length_prior = (len_base ** max(0, hops - 1))
                    path_score = max(0.0, base_ps * length_prior)

            path_scores_local[key] = path_score

            cos1 = _cos(intent_vec, getattr(candF, "vec", []))
            cos2 = _cos(getattr(candF, "vec", []), getattr(money_field, "vec", [])) if money_field else 0.0
            cos_ctx = (cos1 + cos2) / (2.0 if money_field else 1.0)

            role_coh = _role_prob(candF, role)

            local_bonus = metric_bonus if metric_entity and sf.entity == metric_entity else 0.0

            score = (w_path * max(min_path, path_score)) + (w_cos * cos_ctx) + (w_role * role_coh) + local_bonus
            why_dict = {
                "PathScore": float(path_score),
                "CosineContext": float(cos_ctx),
                "RoleCoherence": float(role_coh),
                "LocalBonus": float(local_bonus),
                "weights": {"path": w_path, "cosine": w_cos, "role": w_role}
            }
            if debug_meta:
                why_dict["timestampInfo"] = debug_meta
            scored.append({
                "key": key,
                "sf": sf,
                "score": float(score),
                "finalScore": float(score),  # Initialize finalScore (may be updated later for rule overrides)
                "reason": "normal",  # Default reason (may be updated later)
                "why": why_dict
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        winner = scored[0] if scored else None
        winner_idx = 0

        # Use unknown_backoff cascade if no clear winner or as fallback when scores are too close
        if not winner or (winner and len(scored) > 1 and scored[0]["score"] - scored[1]["score"] < 0.05):
            # Convert scored dicts to ScoredField objects for backoff
            candidates = [s["sf"] for s in scored]
            ctx = {
                "pivot": pivot,
                "salience": salience_map,
                "pathScores": path_scores_local,
                "emergent": active_principles.get("emergent", {})
            }
            backoff_winner = _unknown_backoff(role, candidates, ctx)
            if backoff_winner:
                # Find the corresponding scored entry
                backoff_key = f"{backoff_winner.entity}.{backoff_winner.name}"
                winner = next((s for s in scored if s["key"] == backoff_key), winner)
                if winner:
                    winner_idx = next((i for i, s in enumerate(scored) if s["key"] == backoff_key), 0)

        # Apply timestamp override if gap is below threshold
        rule_override_applied = False
        if role == "timestamp" and pivot and winner and len(scored) > 0:
            pivot_entity = getattr(pivot, "entity", None) if pivot else None
            if pivot_entity:
                # Find orders.created_at (or same-entity timestamp) candidate
                orders_created_at = None
                orders_created_at_entity = None
                for cand in scored:
                    key = cand["key"]
                    if "." in key:
                        ent, field = key.split(".", 1)
                        if ent == pivot_entity and "created_at" in field:
                            orders_created_at = cand
                            orders_created_at_entity = ent
                            break
                
                if orders_created_at:
                    best = winner
                    gap = best["score"] - orders_created_at["score"]
                    th = float(_CONFIG.get("fill", {}).get("conflict", {}).get("preferSameEntityTimestampDelta", 0.02))
                    if gap < th:
                        winner = orders_created_at
                        winner_idx = next((i for i, s in enumerate(scored) if s["key"] == orders_created_at["key"]), 0)
                        orders_created_at["finalScore"] = orders_created_at["score"] + th
                        orders_created_at["reason"] = "preferSameEntityTimestamp"
                        rule_override_applied = True
                    else:
                        best["finalScore"] = best["score"]
                        if "reason" not in best:
                            best["reason"] = "bestScore"

        # Build why dict with preferSameEntityTimestampDelta if timestamp override was applied
        why_dict = (winner["why"] if winner else {}).copy()
        if role == "timestamp" and pivot:
            th = float(_CONFIG.get("fill", {}).get("conflict", {}).get("preferSameEntityTimestampDelta", 0.02))
            why_dict["preferSameEntityTimestampDelta"] = th
        
        # Guard against empty scored list or invalid winner_idx
        if not scored or winner_idx >= len(scored):
            # Edge case - shouldn't happen but be safe
            conflicts.append({
                "slot": role,
                "candidates": [],
                "resolution": None,
                "selectedBy": "none",
                "decisionScore": 0.0,
                "bestNumericScore": 0.0,
                "why": why_dict or {},
                "scores": []
            })
        else:
            # Package conflict with transparency fields
            conflict = _package_conflict(role, scored, winner_idx, why_dict, rule_override_applied)
            conflicts.append(conflict)

        if pivot and getattr(pivot, "roleTop", None) == role and winner and f"{pivot.entity}.{pivot.name}" != winner["key"]:
            new_target = winner["key"]

    return conflicts, new_target

def _path_score(iem: IEMIndex, intent_vec: list[float],
                edges: list[tuple[str,str,str,str,str]],
                cfg: dict | None = None) -> float:
    """
    PathScore™ = blend * avgCos + (1-blend) * avgPrior, then * lengthPrior
      - avgCos: mean over edges of ((cos(src) + cos(dst))/2, clamped >= 0)
      - avgPrior: mean over edges of (idPrior_if_*_id + fkBonus_if_fk_edge)
      - lengthPrior: (lengthPriorBase)^(hops-1)   (edges == hops)
    """
    ps = _CONFIG.get("pathScoring", {})
    if cfg:
        ps = {**ps, **cfg}

    id_prior = float(ps.get("idPrior", 0.10))
    fk_bonus = float(ps.get("fkBonus", 0.05))
    len_base = float(ps.get("lengthPriorBase", 0.92))
    w_cos    = float(ps.get("cosineWeight", 0.75))
    # new, schema-aware pieces
    pk_fk_bonus = float(ps.get("pkToFkBonus", 0.00))
    fk_pk_bonus = float(ps.get("fkToPkBonus", 0.00))
    bridge_pen  = float(ps.get("bridgePenalty", 0.00))
    hub_pen     = float(ps.get("hubPenalty", 0.00))
    hub_thresh  = int(ps.get("hubDegreeThreshold", 8))
    fanout_pen  = float(ps.get("fanoutPenalty", 0.00))

    if not edges:
        return 1.0

    name_map = {(f.entity, f.name): f for f in iem.fields}
    # degree map for hub/fanout heuristics
    deg: dict[str,int] = defaultdict(int)
    for (e, lst) in _build_join_graph(iem).items():
        deg[e] = len(lst)

    cos_terms = []
    prior_terms = []

    for (srcE, srcF, dstE, dstF, why) in edges:
        a = name_map.get((srcE, srcF))
        b = name_map.get((dstE, dstF))

        if a and b:
            # cache per (intent_id≈len, entity, field)
            k1 = (len(intent_vec), srcE, srcF)
            k2 = (len(intent_vec), dstE, dstF)
            if k1 not in _COS_CACHE:
                _COS_CACHE[k1] = max(0.0, _cosine(intent_vec, a.vec))
            if k2 not in _COS_CACHE:
                _COS_CACHE[k2] = max(0.0, _cosine(intent_vec, b.vec))
            ca = _COS_CACHE[k1]
            cb = _COS_CACHE[k2]
            cos_terms.append((ca + cb) / 2.0)
        else:
            cos_terms.append(0.0)

        p = 0.0
        if (srcF.endswith("_id") or dstF.endswith("_id")):
            p += id_prior
        if why in {"fk", "fk_rev"}:
            p += fk_bonus
            # direction-aware nudges
            if why == "fk":      # forward PK->FK
                p += pk_fk_bonus
            elif why == "fk_rev":  # reverse FK->PK
                p += fk_pk_bonus
        # fanout / hub penalties
        if deg.get(dstE,0) >= hub_thresh:
            p -= hub_pen
        if fanout_pen:
            p -= fanout_pen * max(0, deg.get(dstE,0) - 1)
        prior_terms.append(p)

    avg_cos = sum(cos_terms) / len(cos_terms) if cos_terms else 0.0
    avg_pri = sum(prior_terms) / len(prior_terms) if prior_terms else 0.0

    raw = (w_cos * avg_cos) + ((1.0 - w_cos) * avg_pri)

    hops = len(edges)
    length_prior = (len_base ** max(0, hops - 1))

    score = float(raw * length_prior)
    # simple "bridge" penalty if the middle of a 3-hop path is very high-degree
    if bridge_pen and hops >= 3:
        mid = edges[hops//2][2]  # dstEntity of middle hop
        if deg.get(mid,0) >= hub_thresh:
            score -= bridge_pen
    return score