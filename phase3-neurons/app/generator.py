from typing import Dict, Any, List, Tuple
import os, json, re
from .iem import IEMIndex

# Allowed roles for group-by pivots (never id or timestamp)
PIVOT_OK = ("category", "geo", "text")

def _best_dim(iem: IEMIndex, entity: str) -> Tuple[str, float] | None:
    """
    Pick a group-by dimension in (category, geo, text) order.
    """
    choices = []
    for role in PIVOT_OK:
        cand = _top_field_for_role(iem, entity, role)
        if cand and cand[1] > 0:
            choices.append((role, cand[0], cand[1]))
    if not choices:
        return None
    _, name, w = max(choices, key=lambda x: x[2])
    return (name, w)

def _top_field_for_role(iem: IEMIndex, entity: str, role: str) -> Tuple[str, float] | None:
    best = None
    best_w = -1.0
    for f in iem.fields:
        if f.entity != entity:
            continue
        w = (f.role or {}).get(role, 0.0)
        # if your IEM adds roleTop, a tiny fallback weight:
        if w == 0 and getattr(f, "role", None) and isinstance(f.role, dict) and f.role.get(role, 0) == 0:
            pass
        if w > best_w:
            best_w = w
            best = f.name
    return (best, best_w) if best else None

def _any_field_for_role(iem: IEMIndex, role: str) -> Tuple[str,str,float] | None:
    """Return (entity, name, weight) highest scoring for role across schema."""
    best = None; best_e = None; best_w = -1.0
    for f in iem.fields:
        w = (f.role or {}).get(role, 0.0)
        if w > best_w:
            best, best_e, best_w = f.name, f.entity, w
    return (best_e, best, best_w) if best else None

def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9\-]+", "-", s.lower())
    return re.sub(r"-+", "-", s).strip("-")

def _mk_iql_topk(entity:str, money:str, dim:str, k:int=10) -> Dict[str,Any]:
    return {
        "ask":"top_k",
        "metric": {"op":"sum", "target": f"{entity}.{money}"},
        "target": f"{entity}.{dim}",
        "timeWindow": {"type":"relative","period":"last_30d"},
        "groupBy": [f"{entity}.{dim}"],
        "orderBy": [{"expr": "metric", "dir": "desc"}],
        "limit": k
    }

def _mk_iql_trend(entity:str, money:str, ts:str) -> Dict[str,Any]:
    return {
        "ask":"trend",
        "metric": {"op":"sum", "target": f"{entity}.{money}"},
        "timeWindow": {"type":"relative","period":"last_30d"},
        "bucket": {"on": f"{entity}.{ts}", "grain": "day"},
        "orderBy": [{"expr": "bucket", "dir":"asc"}]
    }

def _mk_iql_compare(entity:str, money:str, ts:str) -> Dict[str,Any]:
    return {
        "ask":"compare",
        "metric": {"op":"sum", "target": f"{entity}.{money}"},
        "comparePeriods": [
            {"type":"relative","period":"this_month"},
            {"type":"relative","period":"last_month"}
        ],
        "where": [{"op": "is_not_null", "field": f"{entity}.{ts}"}]
    }

def _mk_iql_topk_qty(entity:str, qty:str, dim:str, k:int=10) -> Dict[str,Any]:
    return {
        "ask":"top_k",
        "metric": {"op":"sum", "target": f"{entity}.{qty}"},
        "target": f"{entity}.{dim}",
        "timeWindow": {"type":"relative","period":"last_30d"},
        "groupBy": [f"{entity}.{dim}"],
        "orderBy": [{"expr":"metric","dir":"desc"}],
        "limit": k
    }

def _mk_iql_topk_geo(entity:str, money:str, geo:str, k:int=10) -> Dict[str,Any]:
    return {
        "ask":"top_k",
        "metric": {"op":"sum", "target": f"{entity}.{money}"},
        "target": f"{entity}.{geo}",
        "timeWindow": {"type":"relative","period":"last_30d"},
        "groupBy": [f"{entity}.{geo}"],
        "orderBy": [{"expr":"metric","dir":"desc"}],
        "limit": k
    }

def _title(t:str) -> str:
    return t[0].upper()+t[1:]

def _is_unsafe_pivot(field_name: str) -> bool:
    fn = field_name.lower()
    return fn in {"id","created_at","updated_at","timestamp"} or fn.endswith("_id")

def _is_safe_item(item: Dict[str,Any]) -> bool:
    group_by = item.get("iql", {}).get("groupBy", [])
    if not group_by: 
        return True  # trend/compare don't group
    f = group_by[0].split(".")[-1] if "." in group_by[0] else group_by[0]
    return not _is_unsafe_pivot(f)

def _backfill(iem: IEMIndex, entity: str, per_entity: int, items: List[Dict[str,Any]],
              money: tuple|None, ts: tuple|None, cat: tuple|None, geo: tuple|None, qty: tuple|None) -> List[Dict[str,Any]]:
    """
    Fill shortfall with safe alternates to reach per_entity.
    Priority:
      1) extra trend/compare variants (if money+ts)
      2) top_k by geo/category/text using qty if money missing
      3) if all else fails, duplicate trend with a different grain (week) tagged 'variant'
    """
    need = max(0, per_entity - len(items))
    if need == 0: 
        return items

    def push(title, intent, iql):
        items.append({"title": title, "entity": entity, "intent": intent, "iql": iql, "variant": True})

    # 1) more time variants
    if need and money and ts:
        # weekly trend
        iql = _mk_iql_trend(entity, money[0], ts[0]).copy()
        iql["bucket"]["grain"] = "week"
        push(_title(f"{entity}: revenue trend (weekly, 12w)"), 
             {"ask":"trend","metric":{"op":"sum","target":f"{entity}.{money[0]}"}}, iql)
        need = max(0, per_entity - len(items))

    # 2) top_k by geo/category/text using qty
    if need and qty:
        dim_any = _best_dim(iem, entity)
        if dim_any:
            iql = _mk_iql_topk_qty(entity, qty[0], dim_any[0], 10)
            push(_title(f"{entity}: top {dim_any[0]} by units (30d)"),
                 {"ask":"top_k","metric":{"op":"sum","target":f"{entity}.{qty[0]}"},"target":f"{entity}.{dim_any[0]}"},
                 iql)
            need = max(0, per_entity - len(items))

    # 3) final nops – duplicate safe compare (if available)
    if need and money and ts:
        iql = _mk_iql_compare(entity, money[0], ts[0])
        push(_title(f"{entity}: revenue compare (QoQ proxy: this vs last month)"),
             {"ask":"compare","metric":{"op":"sum","target":f"{entity}.{money[0]}"}}, iql)

    return items[:per_entity]

def generate_for_entity(iem:IEMIndex, entity:str, per_entity:int=5, *, strict: bool=True) -> Tuple[List[Dict[str,Any]], Dict[str,bool]]:
    # pick candidates
    money = _top_field_for_role(iem, entity, "money")
    ts    = _top_field_for_role(iem, entity, "timestamp")
    cat   = _top_field_for_role(iem, entity, "category")
    geo   = _top_field_for_role(iem, entity, "geo")
    qty   = _top_field_for_role(iem, entity, "quantity")
    dim_any = _best_dim(iem, entity)

    cover = {
        "entity": entity,
        "money": bool(money and money[1] > 0),
        "timestamp": bool(ts and ts[1] > 0),
        "category": bool(cat and cat[1] > 0),
        "geo": bool(geo and geo[1] > 0),
        "quantity": bool(qty and qty[1] > 0),
    }

    items: List[Dict[str,Any]] = []
    # 1) top_k by category (or safe fallback)
    if money and (cat or dim_any):
        dim_name = (cat[0] if cat else dim_any[0])
        items.append({
            "title": _title(f"{entity}: top {dim_name} by revenue (30d)"),
            "entity": entity,
            "intent": {"ask":"top_k","metric":{"op":"sum","target":f"{entity}.{money[0]}"},"target":f"{entity}.{dim_name}"},
            "iql": _mk_iql_topk(entity, money[0], dim_name, 10)
        })
    # 2) trend revenue daily (30d)
    if money and ts:
        items.append({
            "title": _title(f"{entity}: revenue trend (daily, 30d)"),
            "entity": entity,
            "intent": {"ask":"trend","metric":{"op":"sum","target":f"{entity}.{money[0]}"}},
            "iql": _mk_iql_trend(entity, money[0], ts[0])
        })
    # 3) compare this vs last month
    if money and ts:
        items.append({
            "title": _title(f"{entity}: revenue compare (this vs last month)"),
            "entity": entity,
            "intent": {"ask":"compare","metric":{"op":"sum","target":f"{entity}.{money[0]}"}},
            "iql": _mk_iql_compare(entity, money[0], ts[0])
        })
    # 4) top_k by geo
    if money and geo:
        items.append({
            "title": _title(f"{entity}: top {geo[0]} by revenue (30d)"),
            "entity": entity,
            "intent": {"ask":"top_k","metric":{"op":"sum","target":f"{entity}.{money[0]}"},"target":f"{entity}.{geo[0]}"},
            "iql": _mk_iql_topk_geo(entity, money[0], geo[0], 10)
        })
    # 5) top_k by quantity with safe dimension
    if qty and (cat or dim_any):
        dim_name = (cat[0] if cat else dim_any[0])
        items.append({
            "title": _title(f"{entity}: top {dim_name} by units (30d)"),
            "entity": entity,
            "intent": {"ask":"top_k","metric":{"op":"sum","target":f"{entity}.{qty[0]}"},"target":f"{entity}.{dim_name}"},
            "iql": _mk_iql_topk_qty(entity, qty[0], dim_name, 10)
        })

    # strict filter (default): drop unsafe pivots
    if strict:
        items = [it for it in items if _is_safe_item(it)]

    # backfill to reach per_entity, keeping safety
    if len(items) < per_entity:
        items = _backfill(iem, entity, per_entity, items, money, ts, cat, geo, qty)

    # attach filenames
    for it in items[:per_entity]:
        it["file"] = f"{_slug(it['title'])}.iql.json"
    return items[:per_entity], cover

def generate_all(iem:IEMIndex, per_entity:int=5, *, strict: bool=True) -> Tuple[List[Dict[str,Any]], List[Dict[str,bool]]]:
    ents = sorted({e.name for e in iem.entities}) if getattr(iem,"entities",None) else sorted({f.entity for f in iem.fields})
    all_items: List[Dict[str,Any]] = []
    coverage: List[Dict[str,bool]] = []
    for ent in ents:
        items, cover = generate_for_entity(iem, ent, per_entity, strict=strict)
        all_items.extend(items)
        coverage.append(cover)
    return all_items, coverage

def write_examples(items: List[Dict[str,Any]], out_dir:str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for it in items:
        p = os.path.join(out_dir, it["file"])
        with open(p, "w") as f:
            json.dump({"title": it["title"], "entity": it["entity"], "intent": it["intent"], "iql": it["iql"]}, f, indent=2)
        n += 1
    return n

def write_coverage(coverage: List[Dict[str,bool]], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "_coverage.json")
    with open(path, "w") as f:
        json.dump({"entities": coverage}, f, indent=2)
    return path
