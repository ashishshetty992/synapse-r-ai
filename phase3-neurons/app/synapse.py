import math, json
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .models import MatchResponse, ScoredField, ScoredEntity
from .iem import IEMIndex

# runtime knobs with safe defaults (fallback only)
_CONFIG = {
    "role_keywords": {},   # filled by set_role_config
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
    },
    "logging": {
        "decisionsEnabled": True,
        "path": "./logs/decisions.jsonl.gz",
    },
}

def set_role_config(keywords: dict[str, list[str]] | None = None):
    if keywords: 
        _CONFIG["role_keywords"] = {k: list(v) for k, v in keywords.items()}
        print(f"DEBUG: set_role_config called with {len(keywords)} roles")

def set_shaping_weights(weights: dict[str, float] | None = None):
    if weights: 
        _CONFIG["shaping_weights"].update({k: float(v) for k, v in weights.items()})
        print(f"DEBUG: set_shaping_weights called with {len(weights)} weights")

def set_entity_blend(entity_weight: float, field_weight: float, entity_alias_alpha: float):
    _CONFIG["entity_blend"] = {
        "entityWeight": float(entity_weight),
        "fieldWeight": float(field_weight),
        "entityAliasAlpha": float(entity_alias_alpha),
    }
    print(f"DEBUG: set_entity_blend called with entity_weight={entity_weight}, field_weight={field_weight}, entity_alias_alpha={entity_alias_alpha}")

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
    _CONFIG["pathScoring"].update({k: float(v) for k, v in d.items()
                                   if k in ("idPrior","fkBonus","lengthPriorBase","cosineWeight")})

def set_logging_cfg(d: dict | None):
    if not d: return
    cur = _CONFIG["logging"]
    if "decisionsEnabled" in d: cur["decisionsEnabled"] = bool(d["decisionsEnabled"])
    if "path" in d and d["path"]: cur["path"] = str(d["path"])

def set_path_scoring_config(d: dict | None):
    """
    Update PathScore™ knobs from shaping.json["pathScoring"].
    Keys: idPrior, fkBonus, lengthPriorBase, cosineWeight
    """
    if not d: return
    cur = _CONFIG["pathScoring"]
    for k in ("idPrior", "fkBonus", "lengthPriorBase", "cosineWeight"):
        if k in d and d[k] is not None:
            cur[k] = float(d[k])
    print(f"DEBUG: set_path_scoring_config -> {cur}")


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
    # both are already l2-normalized; dot product is cosine
    return sum(x * y for x, y in zip(a, b))


def _role_prior_from_centroids(intent_vec: List[float], role_centroids: Dict[str, List[float]]) -> Dict[str, float]:
    # cosine vs per-role centroid; both already (or treated as) L2
    scores = {}
    for role, cvec in role_centroids.items():
        scores[role] = _cosine(intent_vec, cvec)
    # clamp negatives to 0 for a prob-like distribution
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
    # normalize
    s = sum(scores.values()) or 1.0
    return {k: v / s for k, v in scores.items()}


def match_candidates(
    iem: IEMIndex,
    intent_vec: List[float],
    top_k_fields: int = 8,
    top_k_entities: int = 5,
    role_alpha: Optional[float] = None,
    intent_obj: Dict[str, Any] | None = None,
    # + NEW: shaping parameters for A/B tuning
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

    # --- 1) keyword prior (existing) ---
    if intent_obj:
        keyword_prior = _role_prior_from_intent(intent_obj)
        print(f"DEBUG: keyword_prior from intent_obj: {keyword_prior}")
    else:
        intent_signature = " ".join([str(round(x, 3)) for x in intent_vec[:32]])
        keyword_prior = _guess_intent_role_tokens(intent_signature)
        print(f"DEBUG: keyword_prior from signature: {keyword_prior}")

    # --- 2) centroid prior (NEW, schema-adaptive) ---
    centroid_prior: Dict[str, float] = {}
    if getattr(iem, "role_centroids", None):
        print(f"DEBUG: role_centroids available: {list(iem.role_centroids.keys())}")
        # cosine with each centroid; clamp negatives to 0 then normalize
        sims = {}
        a = intent_vec
        for r, cvec in iem.role_centroids.items():
            b = cvec
            # both are l2-normalized → dot is cosine; be defensive anyway
            dot = sum(x*y for x, y in zip(a, b))
            sims[r] = max(0.0, float(dot))
        s = sum(sims.values()) or 1.0
        centroid_prior = {k: v/s for k, v in sims.items()}
        print(f"DEBUG: centroid_prior computed: {centroid_prior}")
    else:
        print("DEBUG: No role_centroids available in IEM")

    # --- 3) blend both priors (NEW) ---
    gamma = _CONFIG["centroid_gamma"]
    role_prior = _blend_priors(keyword_prior, centroid_prior, gamma=gamma)  # 60% schema-adaptive
    print(f"DEBUG: role_prior blended with gamma={gamma}: {role_prior}")

    # normalize target(s) to (entity, field) list
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

        # role bonus: use top role vs prior
        top_role = max(f.role.items(), key=lambda kv: kv[1])[0] if f.role else None
        role_bonus = role_prior.get(top_role or "", 0.0) * role_alpha

        # --- ask-aware shaping & explicit target boost ---
        ask = (intent_obj or {}).get("ask")
        metric = (intent_obj or {}).get("metric", {})
        metric_op = (metric or {}).get("op")

        # --- alias/target bonus (robust) ---
        norm = lambda s: (s or "").lower().strip()
        def variants(s: str) -> set[str]:
            s = norm(s)
            return {s, s.replace("_", " "), s.replace(" ", "_")}

        target_field_tokens: set[str] = set()
        for (_, b) in target_pairs:
            target_field_tokens |= variants(b)

        is_target_exact = (f.entity, f.name) in target_pairs
        aliases = (f.aliases or []) + [f.name]   # include canonical as an alias
        has_alias_hit = any(
            (norm(al) in target_field_tokens) or (variants(al) & target_field_tokens)
            for al in aliases
        )

        # weights (from configurable SHAPING_WEIGHTS or provided parameters)
        sw = _CONFIG["shaping_weights"]
        alias_alpha_val  = alias_alpha if alias_alpha is not None else sw["alias_alpha"]
        shape_alpha_val   = shape_alpha if shape_alpha is not None else sw["shape_alpha"]
        shape_beta_val    = shape_beta if shape_beta is not None else sw["shape_beta"]
        metric_alpha_val  = metric_alpha if metric_alpha is not None else sw["metric_alpha"]

        # alias boost (explicit target mention)
        alias_bonus = alias_alpha_val if (is_target_exact or has_alias_hit) else 0.0

        # ask-aware shaping
        role_shape = 0.0
        if ask == "top_k" and metric_op in {"count", "distinct_count"}:
            if top_role in {"geo", "category", "text"}:
                role_shape += shape_alpha_val
            if top_role in {"money"}:
                role_shape -= shape_beta_val

        # metric-aware shaping
        if metric_op in {"sum", "avg", "mean", "median"} and top_role == "money":
            role_shape += metric_alpha_val

        # final score (mix multiplicative base + additive nudges)
        score = base * (1.0 + role_bonus) + alias_bonus + role_shape
        scored_fields.append(ScoredField(entity=f.entity, name=f.name, score=score, roleTop=top_role))

    # rank fields
    scored_fields.sort(key=lambda x: x.score, reverse=True)
    top_fields = scored_fields[:top_k_fields]

    # define entity blend config once, always available
    eb = _CONFIG["entity_blend"]

    # rank entities using stored entity embeddings if available
    top_entities: List[ScoredEntity] = []
    if getattr(iem, "entities", None):
        # pre-collect top field scores per entity (use the same scored_fields you already have)
        per_ent_fields = defaultdict(list)
        for sf in scored_fields[: max(64, top_k_fields * 4)]:
            per_ent_fields[sf.entity].append(sf.score)

        # small target bonus if entity is explicitly in targetPairs
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
        # fallback: avg of top field scores considered (existing behavior)
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
            "keywordPrior": keyword_prior,        # + NEW
            "centroidPrior": centroid_prior,      # + NEW
            "blendWeights": {"keyword": round(1-gamma,2), "centroid": round(gamma,2)},
            "entityBlend": eb,
            "roleAlphaUsed": role_alpha,
            "shapingWeights": {                   # + NEW: show actual weights used
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


# ---------- Neuron-3 helpers: Slot Fill + AlignPlus + PathScore ----------

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
    # Coverage: ask + metric.op + target (+ timeWindow if present)
    slots = 0; hit = 0
    for key in ["ask", "metric", "target"]:
        slots += 1
        if key in intent and intent[key]:
            hit += 1
    # optional bonus slot
    if "timeWindow" in intent: slots += 1; hit += 1
    return hit / max(1, slots)


def _off_schema_rate(tokens: list[str], mapped: set[str]) -> float:
    if not tokens: return 1.0
    bad = sum(1 for t in tokens if t not in mapped)
    return bad / len(tokens)


def compute_alignplus(Abase: float, intent: Dict[str, Any],
                      tokens: list[str], mapped_terms: set[str],
                      shaping_cfg: dict = None) -> dict:
    cov = _coverage_score(intent)

    # NEW: make OSR on "no text" configurable via shaping.json
    align_cfg = (shaping_cfg or {}).get("alignPlus", {}) if shaping_cfg else {}
    osr_zero_when_no_text = bool(align_cfg.get("osrZeroWhenNoText", True))
    if not tokens and osr_zero_when_no_text:
        osr = 0.0
    else:
        osr = _off_schema_rate(tokens, mapped_terms)

    Aplus = float(Abase * (1.0 - osr) * (0.5 + 0.5 * cov))
    return {
        "Abase": float(Abase),
        "Coverage": float(cov),
        "OffSchemaRate": float(osr),
        "Aplus": float(Aplus),
    }


def _build_join_graph(iem: IEMIndex) -> dict[str, list[tuple[str,str,str,str]]]:
    """Return adjacency: entity -> list of (srcField, dstEntity, dstField, why)"""
    from collections import deque
    adj: dict[str, list[tuple[str,str,str,str]]] = defaultdict(list)
    for j in (getattr(iem, "joins", []) or []):
        adj[j.srcEntity].append((j.srcField, j.dstEntity, j.dstField, "fk"))
        # also add reverse (assume bidirectional traversal is possible)
        adj[j.dstEntity].append((j.dstField, j.srcEntity, j.srcField, "fk_rev"))
    return adj


def _bfs_paths(adj: dict[str, list[tuple[str,str,str,str]]],
               start: str, goal: str, max_hops: int = 2) -> list[list[tuple[str,str,str,str]]]:
    """BFS over entities; edges carry (srcField, dstEntity, dstField, why)"""
    from collections import deque
    if start == goal: return [[]]
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
    return paths


def _top_role_of_field(f) -> str:
    if not getattr(f, "role", None): return "unknown"
    return max(f.role.items(), key=lambda kv: kv[1])[0]

def _field_map(iem) -> dict:
    return {(f.entity, f.name): f for f in iem.fields}

def _cos(a: list[float], b: list[float]) -> float:
    return sum(x*y for x, y in zip(a, b)) if a and b else 0.0

def _role_prob(f, role: str) -> float:
    if not getattr(f, "role", None): return 0.0
    return float(f.role.get(role, 0.0))

def _metric_entity_from_intent(intent: dict[str, Any]) -> str | None:
    tgt = (intent or {}).get("target")
    if isinstance(tgt, str) and "." in tgt:
        return tgt.split(".")[0]
    return None

# def _canonical_slot_name(role_top: str, field_name: str) -> str:
#     """Map role+name to generic slot id (e.g., any '...city' under geo → 'city')."""
#     fname = field_name.lower()
#     if role_top == "geo":
#         if fname.endswith("city"):
#             return "city"
#         if fname.endswith("state"):
#             return "state"
#         if fname.endswith("country"):
#             return "country"
#         return "geo"
#     if role_top == "timestamp":
#         if "date" in fname: return "date"
#         return "timestamp"
#     if role_top == "id": return "id"
#     if role_top == "money": return "money"
#     return role_top or "unknown"

# def _iem_field_lookup(iem: IEMIndex) -> dict[tuple[str,str], Any]:
#     return {(f.entity, f.name): f for f in iem.fields}

# def _join_distance(iem: IEMIndex, start_entity: str, target_entity: str, max_hops: int = 3) -> int:
#     """BFS over entity graph; returns hop count or large number if unreachable."""
#     adj = _build_join_graph(iem)
#     if start_entity == target_entity: return 0
#     from collections import deque
#     q = deque([(start_entity, 0)])
#     seen = {start_entity}
#     while q:
#         e, d = q.popleft()
#         if d >= max_hops: 
#             continue
#         for (_, nxt, _, _) in adj.get(e, []):
#             if nxt in seen: 
#                 continue
#             if nxt == target_entity: 
#                 return d + 1
#             seen.add(nxt)
#             q.append((nxt, d + 1))
#     return 1_000_000  # unreachable sentinel

# def _name_similarity(a: str, b: str) -> float:
#     """Lightweight name sim for tie-breaks; 1 if exact, else soft by token overlap."""
#     import re
#     a, b = (a or "").lower(), (b or "").lower()
#     if a == b: return 1.0
#     at = set(re.findall(r"[a-z0-9]+", a))
#     bt = set(re.findall(r"[a-z0-9]+", b))
#     if not at or not bt: return 0.0
#     return len(at & bt) / len(at | bt)

# def _resolve_conflicts(
#     iem: IEMIndex,
#     pivot_entity: str,
#     candidates: list[ScoredField],
#     intent_obj: dict,
#     prefer_target_entity: bool = True,
# ) -> list:
#     """
#     Group same-slot competitors across entities (e.g., orders.shipping_city vs customer.city),
#     rank by: (1) prefer target-entity, (2) closest join distance to pivot, (3) role weight,
#     (4) base score, (5) name similarity to pivot.
#     """
#     from .models import ConflictNote
#     # group by slot id
#     by_slot: dict[str, list[ScoredField]] = {}
#     for sf in candidates:
#         slot = _canonical_slot_name(sf.roleTop, sf.name)
#         if slot not in by_slot: by_slot[slot] = []
#         by_slot[slot].append(sf)

#     target_ent = None
#     tgt = (intent_obj or {}).get("target")
#     if isinstance(tgt, str) and "." in tgt:
#         target_ent = tgt.split(".")[0]

#     notes: list = []
#     fidx = _iem_field_lookup(iem)

#     for slot, sfs in by_slot.items():
#         # only "conflict" if same-slot appears across >1 distinct entity
#         ents = {sf.entity for sf in sfs}
#         if len(ents) <= 1:
#             continue

#         scored: list[tuple[float, ScoredField]] = []
#         for sf in sfs:
#             pref = 0.0
#             why_bits = []
#             # (1) prefer explicit target entity
#             if prefer_target_entity and target_ent and sf.entity == target_ent:
#                 pref += 0.12
#                 why_bits.append("preferTargetEntity")
#             # (2) closer to pivot entity
#             if pivot_entity:
#                 hops = _join_distance(iem, pivot_entity, sf.entity, max_hops=4)
#                 hop_bonus = max(0.0, (3.0 - min(hops, 3)) * 0.05)  # 0.15 → 0.10 → 0.05 → 0
#                 pref += hop_bonus
#                 if hop_bonus > 0: why_bits.append("closestJoin")
#             # (3) role weight (stored in IEM role map)
#             fmeta = fidx.get((sf.entity, sf.name))
#             role_w = 0.0
#             if fmeta and fmeta.role:
#                 role_w = max(fmeta.role.values())
#                 pref += 0.05 * role_w
#                 if role_w > 0: why_bits.append("roleWeight")
#             # (4) base score as is
#             base = float(sf.score)
#             # (5) name similarity to pivot field (if pivot has a field name)
#             piv_field_name = None
#             if isinstance(tgt, str) and "." in tgt:
#                 piv_field_name = tgt.split(".")[1]
#             if piv_field_name:
#                 pref += 0.03 * _name_similarity(sf.name, piv_field_name)

#             final = base + pref
#             scored.append((final, sf))

#         scored.sort(key=lambda x: x[0], reverse=True)
#         winner = scored[0][1]
#         why_str = "|".join(sorted(set(
#             ["preferTargetEntity"] if (prefer_target_entity and target_ent and winner.entity == target_ent) else []
#         )) or ["composite"])
        
#         note = ConflictNote(
#             slot=slot,
#             candidates=[f"{sf.entity}.{sf.name}" for _, sf in scored],
#             resolution=f"{winner.entity}.{winner.name}",
#             why=why_str,
#             scores={f"{sf.entity}.{sf.name}": round(score, 6) for score, sf in scored}
#         )
#         notes.append(note)

#     return notes

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

    # field lookup
    fmap = {(f.entity, f.name): f for f in iem.fields}
    f = fmap.get((entity, field))
    if not f:
        return {"CosineContext": 0.0, "RoleCoherence": 0.0, "PathScore": 0.0}

    # cosine + role
    cos = max(0.0, sum(x*y for x, y in zip(intent_vec, f.vec)))
    role_top = max(f.role.items(), key=lambda kv: kv[1])[0] if f.role else "unknown"
    role_prob = float((f.role or {}).get(role_top, 0.0))

    # --- join-aware PathScore ---
    # pick a pivot entity from intent (money entity for sum/avg; else explicit target entity)
    pivot_entity = None
    if intent_obj:
        metric_op = ((intent_obj.get("metric") or {}).get("op") or "").lower()
        if metric_op in {"sum", "avg", "mean", "median"}:
            # pick highest "money" field's entity as pivot
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
        # fall back to candidate's entity (no cross-entity path needed)
        pivot_entity = entity

    if pivot_entity == entity:
        # local: give small idPrior bonus if *_id field, else neutral
        id_prior = _CONFIG["pathScoring"]["idPrior"]
        prior = id_prior if field.endswith("_id") else 0.0
        return {"CosineContext": float(cos), "RoleCoherence": role_prob, "PathScore": float(prior)}

    # graph path from pivot -> entity
    adj = _build_join_graph(iem)
    raw_paths = _bfs_paths(adj, start=pivot_entity, goal=entity, max_hops=3)

    if not raw_paths:
        return {"CosineContext": float(cos), "RoleCoherence": role_prob, "PathScore": 0.0}

    # turn edges into a lightweight prior (same knobs as /paths)
    ps_cfg = _CONFIG["pathScoring"]
    id_prior = float(ps_cfg["idPrior"])
    fk_bonus = float(ps_cfg["fkBonus"])
    length_decay = float(ps_cfg["lengthPriorBase"])  # <1 → slight preference to shorter paths
    cos_w = float(ps_cfg["cosineWeight"])

    best = 0.0
    name_map = {(ff.entity, ff.name): ff for ff in iem.fields}

    for edges in raw_paths:
        # edges are (srcE, srcF, dstE, dstF, why)
        hop_priors = []
        hop_cos = []
        for (srcE, srcF, dstE, dstF, why) in edges:
            prior = 0.0
            if srcF.endswith("_id") or dstF.endswith("_id"):
                prior += id_prior
            if "fk" in (why or ""):
                prior += fk_bonus

            a = name_map.get((srcE, srcF))
            b = name_map.get((dstE, dstF))
            if a and b:
                cval = 0.5 * max(0.0, sum(x*y for x, y in zip(intent_vec, a.vec))) \
                     + 0.5 * max(0.0, sum(x*y for x, y in zip(intent_vec, b.vec)))
            else:
                cval = 0.0

            hop_priors.append(prior)
            hop_cos.append(cval)

        # combine per-hop and apply length prior
        path_prior = sum(hop_priors) / max(1, len(hop_priors))
        path_cos = sum(hop_cos) / max(1, len(hop_cos))
        combined = (cos_w * path_cos) + ((1.0 - cos_w) * path_prior)
        combined *= (length_decay ** (len(edges) - 1))
        best = max(best, combined)

    return {"CosineContext": float(cos), "RoleCoherence": role_prob, "PathScore": float(best)}

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
    # config
    cf = _CONFIG["fill"]["conflict"]
    w_path = float(cf.get("pathWeight", 0.5))
    w_cos  = float(cf.get("cosineWeight", 0.35))
    w_role = float(cf.get("roleWeight", 0.15))
    min_path = float(cf.get("minPathScore", 0.02))
    len_base = float(cf.get("lengthPriorBase", 0.92))
    metric_bonus = float(cf.get("preferMetricEntityBonus", 0.03))

    # precompute maps/graph
    name_map = _field_map(iem)
    adj = _build_join_graph(iem)

    # metric context
    metric_entity = _metric_entity_from_intent(intent_obj) or (match_debug.get("topEntities", [{}])[0:1] or [None])[0]
    if isinstance(metric_entity, dict):
        metric_entity = metric_entity.get("entity")
    # pick a metric field vector (best 'money' if available)
    money_field = None
    for sf in raw_fields:
        if getattr(sf, "roleTop", None) == "money":
            money_field = name_map.get((sf.entity, sf.name))
            break

    # group ambiguous slots by role (we target roles that commonly conflict)
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

    for role, ent_map in by_role.items():
        # consider conflicts only if same-role candidates span >=2 entities
        if len(ent_map) < 2:
            continue

        # compute scores per candidate
        scored = []
        for ent, sf in ent_map.items():
            candF = name_map.get((sf.entity, sf.name))
            # PathScore: best path from metric_entity -> candidate.entity
            path_score = 0.0
            if metric_entity:
                if metric_entity == sf.entity:
                    path_score = min(1.0, 0.15)  # treat co-located as light path confidence
                else:
                    raw_paths = _bfs_paths(adj, start=metric_entity, goal=sf.entity, max_hops=3)
                    if raw_paths:
                        base_ps = _path_score(iem, intent_vec, raw_paths[0])
                        hops = len(raw_paths[0])
                        length_prior = (len_base ** max(0, hops - 1))
                        path_score = max(0.0, base_ps * length_prior)

            # CosineContext
            cos1 = _cos(intent_vec, getattr(candF, "vec", []))
            cos2 = _cos(getattr(candF, "vec", []), getattr(money_field, "vec", [])) if money_field else 0.0
            cos_ctx = (cos1 + cos2) / (2.0 if money_field else 1.0)

            # RoleCoherence
            role_coh = _role_prob(candF, role)

            # Small bump if candidate entity == metric entity
            local_bonus = metric_bonus if metric_entity and sf.entity == metric_entity else 0.0

            # Combine
            score = (w_path * max(min_path, path_score)) + (w_cos * cos_ctx) + (w_role * role_coh) + local_bonus
            scored.append({
                "key": f"{sf.entity}.{sf.name}",
                "sf": sf,
                "score": float(score),
                "why": {
                    "PathScore": float(path_score),
                    "CosineContext": float(cos_ctx),
                    "RoleCoherence": float(role_coh),
                    "LocalBonus": float(local_bonus),
                    "weights": {"path": w_path, "cosine": w_cos, "role": w_role}
                }
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        winner = scored[0] if scored else None

        conflicts.append({
            "slot": role,
            "candidates": [s["key"] for s in scored],
            "resolution": (winner["key"] if winner else None),
            "why": (winner["why"] if winner else {}),
            "scores": [{"target": s["key"], "score": s["score"], **s["why"]} for s in scored]
        })

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
    if cfg:  # allow direct override injection
        ps = {**ps, **cfg}

    id_prior = float(ps.get("idPrior", 0.10))
    fk_bonus = float(ps.get("fkBonus", 0.05))
    len_base = float(ps.get("lengthPriorBase", 0.92))
    w_cos   = float(ps.get("cosineWeight", 0.75))

    if not edges:
        return 1.0  # trivial path

    name_map = {(f.entity, f.name): f for f in iem.fields}

    cos_terms = []
    prior_terms = []

    for (srcE, srcF, dstE, dstF, why) in edges:
        a = name_map.get((srcE, srcF))
        b = name_map.get((dstE, dstF))

        if a and b:
            ca = max(0.0, _cosine(intent_vec, a.vec))
            cb = max(0.0, _cosine(intent_vec, b.vec))
            cos_terms.append((ca + cb) / 2.0)
        else:
            # unknown field(s) → conservative cosine
            cos_terms.append(0.0)

        p = 0.0
        if (srcF.endswith("_id") or dstF.endswith("_id")):
            p += id_prior
        if why in {"fk", "fk_rev"}:
            p += fk_bonus
        prior_terms.append(p)

    avg_cos = sum(cos_terms) / len(cos_terms) if cos_terms else 0.0
    avg_pri = sum(prior_terms) / len(prior_terms) if prior_terms else 0.0

    raw = (w_cos * avg_cos) + ((1.0 - w_cos) * avg_pri)

    hops = len(edges)
    length_prior = (len_base ** max(0, hops - 1))

    return float(raw * length_prior)