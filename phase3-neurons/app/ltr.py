# neurons/ltr.py
import math
from typing import Dict, List, Tuple, Iterable

# A "raw path" edge here is a 5-tuple: (srcEntity, srcField, dstEntity, dstField, why)
RawEdge = Tuple[str, str, str, str, str]
RawPath = List[RawEdge]

def build_degree_index(iem) -> Dict[str, int]:
    """
    Degree = unique-neighbor count per entity in the join graph.
    Works off iem.joins which has (srcEntity, srcField, dstEntity, dstField, why='fk'|'fk_rev').
    """
    deg: Dict[str, set] = {}
    for j in (getattr(iem, "joins", []) or []):
        deg.setdefault(j.srcEntity, set()).add(j.dstEntity)
        deg.setdefault(j.dstEntity, set()).add(j.srcEntity)
    return {e: len(neis) for e, neis in deg.items()}

def _dir_counters(edges: RawPath) -> Tuple[int, int]:
    """
    Count directionality by 'why':
      - 'fk'     means we go FK -> PK (child -> parent)
      - 'fk_rev' means we go PK -> FK (parent -> child)
    """
    fk_to_pk = 0
    pk_to_fk = 0
    for (_, _sf, _de, _df, why) in edges:
        if why == "fk":
            fk_to_pk += 1
        elif why == "fk_rev":
            pk_to_fk += 1
    return fk_to_pk, pk_to_fk

def _hub_leaf_counts(edges: RawPath, deg_index: Dict[str, int], hub_thresh: int) -> Tuple[int, int, int]:
    """
    Count how many steps traverse hubs/leaves and simple fan-outs.
    We look at the *source* entity degree for each hop.
    Returns: (hubSteps, leafSteps, fanoutSteps)
    """
    hub_steps = 0
    leaf_steps = 0
    fanout_steps = 0
    for (se, _sf, de, _df, _why) in edges:
        sd = deg_index.get(se, 0)
        dd = deg_index.get(de, 0)
        if sd >= hub_thresh:
            hub_steps += 1
            # if we leave a hub into many branches, treat as potential fan-out
            if sd > dd:
                fanout_steps += 1
        elif sd <= 1:
            leaf_steps += 1
    return hub_steps, leaf_steps, fanout_steps

def _bridge_count(edges: RawPath, deg_index: Dict[str, int]) -> int:
    """
    Very light "bridge" heuristic:
      count interior entities with degree == 1 (i.e., narrow bottlenecks/bridges).
    """
    if not edges:
        return 0
    ents: List[str] = [edges[0][0]] + [e[2] for e in edges]  # start entity + all dsts
    if len(ents) <= 2:
        return 0
    interior = ents[1:-1]
    return sum(1 for e in interior if deg_index.get(e, 0) == 1)

def _unique_entities(edges: RawPath) -> int:
    ents: List[str] = [edges[0][0]] + [e[2] for e in edges]
    return len(set(ents))

def _revisits(edges: RawPath) -> int:
    ents: List[str] = [edges[0][0]] + [e[2] for e in edges]
    seen = set()
    dup = 0
    for e in ents:
        if e in seen:
            dup += 1
        else:
            seen.add(e)
    return dup

def _safe(ps_cfg: Dict, key: str, default: float) -> float:
    try:
        return float(ps_cfg.get(key, default))
    except Exception:
        return float(default)

def _clip01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

def _path_delta_from_features(
    feats: Dict[str, float],
    ps_cfg: Dict
) -> float:
    """
    Compute a *small* additive delta using the extended knobs:
      + pkToFkBonus per PK->FK step (fan-out exploration)
      + fkToPkBonus per FK->PK step (roll-ups to parent)
      - hubPenalty per hub step
      - bridgePenalty per bridge
      - fanoutPenalty per fanout step
    The delta is intentionally bounded so it gently reorders ties or near-ties.
    """
    pk_to_fk_bonus = _safe(ps_cfg, "pkToFkBonus", 0.0)
    fk_to_pk_bonus = _safe(ps_cfg, "fkToPkBonus", 0.0)
    hub_pen        = _safe(ps_cfg, "hubPenalty", 0.0)
    bridge_pen     = _safe(ps_cfg, "bridgePenalty", 0.0)
    fanout_pen     = _safe(ps_cfg, "fanoutPenalty", 0.0)

    delta = 0.0
    delta += feats["pkToFkSteps"] * pk_to_fk_bonus
    delta += feats["fkToPkSteps"] * fk_to_pk_bonus
    delta -= feats["hubSteps"]     * hub_pen
    delta -= feats["bridges"]      * bridge_pen
    delta -= feats["fanoutSteps"]  * fanout_pen

    # Keep perturbation modest relative to upstream PathScore range [~0..1]
    # Bound to [-0.25, +0.25], then shrink a bit so we don't dominate base score
    delta = max(-0.25, min(0.25, delta)) * 0.5
    return float(delta)

def _path_feats(edges: RawPath, deg_index: Dict[str, int], hub_thresh: int | None = None) -> Dict[str, float]:
    """
    Public: used by main.py to emit optional features in debug.pathFeatures
    """
    if not edges:
        return {
            "hops": 0.0, "uniqueEntities": 1.0, "revisits": 0.0,
            "fkToPkSteps": 0.0, "pkToFkSteps": 0.0,
            "hubSteps": 0.0, "leafSteps": 0.0, "fanoutSteps": 0.0,
            "bridges": 0.0, "hubThresholdUsed": float(hub_thresh or 6),
        }
    
    if hub_thresh is None:
        hub_thresh =  _default_hub_thresh(deg_index)
    fk_to_pk, pk_to_fk = _dir_counters(edges)
    hub_steps, leaf_steps, fanout_steps = _hub_leaf_counts(edges, deg_index, hub_thresh)
    feats = {
        "hops": float(len(edges)),
        "uniqueEntities": float(_unique_entities(edges)),
        "revisits": float(_revisits(edges)),
        "fkToPkSteps": float(fk_to_pk),
        "pkToFkSteps": float(pk_to_fk),
        "hubSteps": float(hub_steps),
        "leafSteps": float(leaf_steps),
        "fanoutSteps": float(fanout_steps),
        "bridges": float(_bridge_count(edges, deg_index)),
        "hubThresholdUsed": float(hub_thresh),
    }
    return feats

def _default_hub_thresh(deg_index: Dict[str, int]) -> int:
    if not deg_index:
        return 8
    # Use 80th percentile-ish (quick heuristic) but at least 6
    vals = sorted(deg_index.values())
    idx = max(0, int(0.8 * (len(vals) - 1)))
    return max(6, vals[idx])

def ltr_rerank_paths(
    iem,
    scored_paths: List[Tuple[float, RawPath]],
    ps_cfg: Dict,
    degree_index: Dict[str, int] | None = None
) -> List[Tuple[float, RawPath]]:
    """
    Inputs:
      - scored_paths: [(base_score, raw_edges), ...] where raw_edges is RawPath
      - ps_cfg: pathScoring config dict with extended knobs present
      - degree_index: optional prebuilt degree map
    Returns the same list with updated scores and sorted desc by score.
    """
    if not scored_paths:
        return scored_paths

    deg = degree_index or build_degree_index(iem)
    hub_thresh = int(ps_cfg.get("hubDegreeThreshold", _default_hub_thresh(deg)))

    out: List[Tuple[float, RawPath]] = []
    for base, edges in scored_paths:
        feats = _path_feats(edges, deg, hub_thresh)
        delta = _path_delta_from_features(feats, ps_cfg)
        out.append((float(base + delta), edges))

    out.sort(key=lambda t: t[0], reverse=True)
    return out