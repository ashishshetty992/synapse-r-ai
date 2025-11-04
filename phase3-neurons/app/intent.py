import math, re
from typing import Dict, Any, List, Tuple, Set, Iterable
from collections import defaultdict, Counter

import numpy as np

# ========== Typo-tolerant helpers ==========

_word = re.compile(r"[a-z0-9_]+")

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", s.lower().strip())

def _tokens(s: str) -> list[str]:
    return [m.group(0) for m in _word.finditer(s.lower())]

def _dl_dist(a: str, b: str) -> int:
    """Damerau-Levenshtein distance for typo correction."""
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    # 2 rows (rolling)
    prev = list(range(lb + 1))
    curr = [0]*(lb + 1)
    prev_prev = None
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            curr[j] = min(
                curr[j-1] + 1,           # ins
                prev[j] + 1,             # del
                prev[j-1] + cost,        # sub
            )
            # transposition
            if i > 1 and j > 1 and a[i-2] == b[j-1] and a[i-1] == b[j-2]:
                if prev_prev is not None:
                    curr[j] = min(curr[j], prev_prev[j-2] + 1)
        prev_prev, prev, curr = prev, curr, [0]*(lb + 1)
    return prev[lb]

def _is_fuzzy(a: str, b: str, max_ratio: float = 0.34) -> bool:
    """Check if two strings are similar enough (typo-tolerant)."""
    a, b = _norm(a), _norm(b)
    if not a or not b: return False
    d = _dl_dist(a, b)
    # allow ~1 edit per 3 chars (bounded)
    allowed = max(1, int(round(max(len(a), len(b)) * max_ratio)))
    return d <= allowed

def _has_money_signal(tokens: Iterable[str]) -> bool:
    """Detect money-related keywords (typo-tolerant)."""
    money_words = ["revenue","gmv","sales","amount","price","total"]
    return any(any(_is_fuzzy(t, mw) for mw in money_words) for t in tokens)

def _best_geo_field(iem) -> str | None:
    """Pick field with highest geo role weight."""
    best_f, best_w = None, -1.0
    for f in iem.fields:
        if "geo" in (f.role or {}):
            w = f.role["geo"]
            if w > best_w:
                best_w, best_f = w, f
    return f"{best_f.entity}.{best_f.name}" if best_f else None

def _lower(s: str) -> str:
    return (s or "").lower().strip()

def _normalize_tokens_with_aliases(tokens: List[str], aliases: Dict[str, str]) -> List[str]:
    # token-level alias normalization (e.g., mnth→month, qtr→quarter)
    amap = { _lower(k): _lower(v) for k, v in (aliases or {}).items() }
    return [amap.get(_lower(t), t) for t in tokens]

def _detect_time_tags(fixed_text: str, tokens: List[str], time_cfg: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """
    Use config/global/tenant time.json:
      - aliases: token-level normalization (mnth→month, qtr→quarter, …)
      - relative: phrase→canonical_tag (e.g., "last month"→"last_month")
    Returns list of (target, score, why) to be merged with synonyms/BM25.
    """
    hits: List[Tuple[str, float, str]] = []
    rel_map = { _lower(k): str(v) for k, v in (time_cfg or {}).get("relative", {}).items() }
    aliases = { _lower(k): _lower(v) for k, v in (time_cfg or {}).get("aliases", {}).items() }

    # normalize tokens by aliases and rebuild a space-joined string for phrase checks
    norm_tokens = _normalize_tokens_with_aliases(tokens, aliases)
    search_text = " ".join(norm_tokens).lower()

    # greedy phrase match over configured relative phrases
    for ph, tag in sorted(rel_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        if ph in search_text:
            hits.append((f"@relative.time:period={tag}", 1.15, "synonym"))

    return hits

# shared n-grams
def char_ngrams(s: str, n_min=2, n_max=5):
    s = f"##{s.lower()}##"
    for n in range(n_min, n_max + 1):
        for i in range(len(s) - n + 1):
            yield s[i : i + n]


def _bag_strings(d: Dict[str, Any]) -> str:
    out = []
    out.append(str(d.get("ask", "")))

    def _push(x):
        if isinstance(x, (str, int, float, bool)):
            out.append(str(x))
        elif isinstance(x, dict):
            for k, v in x.items():
                _push(k); _push(v)
        elif isinstance(x, list):
            for v in x:
                _push(v)

    for key in ["target", "targets", "metric", "filters", "timeWindow", "orderBy", "comparePeriods", "select"]:
        if key in d:
            _push(d[key])

    return " ".join(s for s in out if s)


def _build_vocab_from_text(text: str, dim: int = 256) -> List[str]:
    freq = defaultdict(int)
    for g in char_ngrams(text, 2, 5):
        freq[g] += 1
    return [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:dim]]


def _sparse_from_text(text: str) -> Dict[str, float]:
    v = defaultdict(int)
    for g in char_ngrams(text, 2, 5):
        v[g] += 1
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {k: x / n for k, x in v.items()}


def _dense_from_sparse(sparse: Dict[str, float], vocab: List[str]) -> List[float]:
    return [sparse.get(tok, 0.0) for tok in vocab]


def encode_intent_to_vocab(intent: Dict[str, Any], vocab: List[str] | None, return_vocab: bool = False) -> Tuple[List[float], List[str]] | List[float]:
    text = _bag_strings(intent)
    sparse = _sparse_from_text(text)
    if vocab is None:
        vocab = _build_vocab_from_text(text, 256)
    dense = _dense_from_sparse(sparse, vocab)
    # defensive l2
    n = math.sqrt(sum(x * x for x in dense)) or 1.0
    dense = [x / n for x in dense]
    if return_vocab:
        return dense, vocab
    return dense

# ========== SynoMix™ v0 (F6 Formula Lab) ==========

WORD_RE = re.compile(r"[a-z0-9_]+")
SPACE_UNDERSCORE_REPLACERS = [("_", " "), (" ", "_")]

def _norm_soft(s: str) -> str:
    """Softer normalizer for phrase/alias handling (does NOT strip underscores/numbers)."""
    return (s or "").lower().strip()

def _variants(s: str) -> Set[str]:
    s = _norm_soft(s)
    return {s, s.replace("_", " "), s.replace(" ", "_")}

def _tokenize(text: str) -> List[str]:
    return WORD_RE.findall(_norm_soft(text))

def _tokenize_with_phrases(text: str, phrase_set: Set[str]) -> List[str]:
    """
    Capture multiword phrases that exist in synonyms (e.g., 'last month') as single tokens.
    """
    t = _norm_soft(text)
    tokens = []
    used = [False]*len(t)
    # greedy phrase match (simple)
    for ph in sorted(phrase_set, key=lambda x: len(x), reverse=True):
        start = t.find(ph)
        while start != -1:
            end = start + len(ph)
            if not any(used[start:end]):
                tokens.append(ph)
                for i in range(start, end): used[i] = True
            start = t.find(ph, end)
    # add residual word tokens
    residual = "".join(ch if not used[i] else " " for i, ch in enumerate(t))
    tokens.extend(WORD_RE.findall(residual))
    return tokens

def _l2_normalize(vec: List[float]) -> List[float]:
    arr = np.array(vec, dtype=float)
    n = np.linalg.norm(arr)
    if n == 0.0: return vec
    return (arr / n).tolist()

def _iem_alias_docs(iem) -> Dict[str, List[str]]:
    """
    Build doc tokens per (entity.field) from canonical name + aliases (with variants).
    """
    docs = {}
    for f in iem.fields:
        target = f"{f.entity}.{f.name}"
        toks = set()
        # canonical
        toks |= _variants(f.name)
        toks |= _variants(target)
        # aliases
        for al in (f.aliases or []):
            toks |= _variants(al)
        # split into words
        wordified = []
        for tt in toks:
            wordified.extend(WORD_RE.findall(tt))
        docs[target] = wordified
    return docs

def _bm25_build(docs: Dict[str, List[str]]) -> Tuple[Dict[str, Counter], Dict[str, int], float, Dict[str, int]]:
    """
    Prepare BM25 statistics: term frequency per doc, doc lengths, avgdl, and document frequency per term.
    """
    tf = {doc_id: Counter(tokens) for doc_id, tokens in docs.items()}
    doc_len = {doc_id: sum(c.values()) for doc_id, c in tf.items()}
    avgdl = (sum(doc_len.values()) / max(1, len(doc_len))) if doc_len else 0.0

    df = Counter()
    for doc_id, c in tf.items():
        for term in c.keys():
            df[term] += 1
    return tf, doc_len, avgdl, df

def _bm25_score_query(tokens: List[str], tf: Dict[str, Counter], doc_len: Dict[str, int],
                      avgdl: float, df: Dict[str, int], N: int,
                      k: float = 1.2, b: float = 0.75) -> Dict[str, float]:
    scores = defaultdict(float)
    for term in tokens:
        n = df.get(term, 0)
        if n == 0: 
            continue
        idf = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
        for doc_id, cnts in tf.items():
            f = cnts.get(term, 0)
            if f == 0: continue
            dl = doc_len[doc_id]
            denom = f + k * (1 - b + b * (dl / (avgdl or 1.0)))
            scores[doc_id] += idf * ((f * (k + 1)) / (denom or 1.0))
    return scores

def _synonym_candidates(tokens: List[str], synonyms: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    """
    From synonyms.json produce (target, score, why) triples.
    """
    # flatten synonyms: key_phrase -> [targets...]
    flat = {}
    phrase_set = set()
    for _, mp in (synonyms or {}).items():
        for k, vs in (mp or {}).items():
            kk = _norm_soft(k)
            flat[kk] = list(vs)
            if " " in kk: phrase_set.add(kk)

    # match phrases first (tokens may already contain phrases)
    hits = []
    tokset = set(tokens)
    for tk in tokset:
        if tk in flat:
            for tgt in flat[tk]:
                # base weight 1.0; small boost for multi-word phrases
                w = 1.15 if " " in tk else 1.0
                hits.append((tgt, w, "synonym"))
    return hits

def _aggregate_candidates(alias_scores: Dict[str, float],
                          syn_hits: List[Tuple[str, float, str]],
                          topK: int = 8) -> List[Dict[str, Any]]:
    """
    Merge BM25 alias candidates and synonym candidates.
    """
    agg = defaultdict(lambda: {"score": 0.0, "why": set()})
    for tgt, sc in alias_scores.items():
        agg[tgt]["score"] += sc
        agg[tgt]["why"].add("alias")
    for tgt, sc, why in syn_hits:
        agg[tgt]["score"] += sc
        agg[tgt]["why"].add(why)

    ranked = sorted(
        [{"target": t, "score": v["score"], "why": sorted(list(v["why"]))} for t, v in agg.items()],
        key=lambda x: x["score"], reverse=True
    )
    return ranked[:topK]

def _vectorize_targets_with_weights(targets: List[Dict[str, Any]], iem_vocab: List[str],
                                    scale: float = 3.0) -> List[float]:
    """
    Build a vector by repeating target strings proportional to their weights, then
    using the existing encode_intent_to_vocab.
    """
    # Build a pseudo-intent with weighted 'select' content
    select = []
    for item in targets:
        t = item["target"]
        w = max(0.0, float(item["score"]))
        reps = int(round(min(4.0, w * scale)))  # cap repeats for stability
        for _ in range(max(1, reps)):
            select.append(t)
    pseudo = {"ask": "infer", "select": select}
    return encode_intent_to_vocab(pseudo, iem_vocab)

def encode_intent_nl(text: str, iem, synonyms: Dict[str, Any],
                     topK: int = 8, alpha_cosine: float = 0.6,
                     time_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    SynoMix v0: BM25 over alias docs + synonyms + cosine blend.
      - tokens ← phrases from synonyms + word tokens
      - alias_docs ← IEM field names + aliases (+ variants)
      - BM25 scores over alias_docs
      - synonym hits → direct target candidates
      - combine → topAliasHits
      - vec_cosine ← encode_intent_to_vocab on light NL stub (ask/target/select)
      - vec_bm25 ← vectorize weighted targets from topAliasHits
      - vec_final ← normalize(alpha*cos + (1-alpha)*bm25)
    """
    # Build lexicon for typo correction
    lexicon: Set[str] = set()
    for f in iem.fields:
        lexicon.add(_norm_soft(f.name))
        for al in (f.aliases or []):
            lexicon.add(_norm_soft(al))
    # synonyms is a dict[str, dict[str, list[str]]]
    for cat, mp in (synonyms or {}).items():
        for k, vs in (mp or {}).items():
            lexicon.add(_norm_soft(k))
            if isinstance(vs, str):
                lexicon.add(_norm_soft(vs))
            elif isinstance(vs, list):
                for v in vs: 
                    lexicon.add(_norm_soft(v))
    # add time aliases from config
    for t in (time_cfg or {}).get("aliases", {}).values():
        lexicon.add(_norm_soft(str(t)))
    time_terms = ["month","mnth","quarter","qtr","week","wk","yesterday","today","last","this","previous","current"]
    for t in time_terms: lexicon.add(_norm_soft(t))

    # Tokenize and typo-correct
    raw_text = text
    toks = _tokens(raw_text)
    
    corrections: List[Dict[str, str]] = []
    fixed: List[str] = []
    for tok in toks:
        nt = _norm_soft(tok)
        if nt in lexicon:
            fixed.append(tok)
            continue
        # find nearest lexicon term
        best = None
        best_len = 1e9
        for cand in lexicon:
            if abs(len(cand) - len(nt)) > 3:  # cheap prune
                continue
            if _is_fuzzy(nt, cand):
                d = _dl_dist(nt, cand)
                if d < best_len:
                    best_len, best = d, cand
                    if d == 0: break
        if best is not None:
            corrections.append({"from": tok, "to": best, "why": "fuzzy"})
            fixed.append(best)
        else:
            fixed.append(tok)
    
    fixed_text = " ".join(fixed)

    # 1) collect phrases from synonyms + time.json(relative phrases)
    phrase_set = set()
    for _, mp in (synonyms or {}).items():
        for k in (mp or {}).keys():
            if " " in k: phrase_set.add(_norm_soft(k))
    # include time phrases from config (no hardcoding)
    for ph in ((time_cfg or {}).get("relative") or {}).keys():
        if " " in ph: phrase_set.add(_lower(ph))

    tokens = _tokenize_with_phrases(fixed_text, phrase_set)

    # 2) alias docs + BM25
    docs = _iem_alias_docs(iem)
    tf, doc_len, avgdl, df = _bm25_build(docs)
    alias_scores = _bm25_score_query(tokens, tf, doc_len, avgdl, df, N=len(docs))

    # 3) synonyms
    syn_hits = _synonym_candidates(tokens, synonyms)

    # 4) time tags from config (table-driven; replaces hardcoded _detect_period)
    time_hits = _detect_time_tags(fixed_text, tokens, time_cfg or {})
    syn_hits.extend(time_hits)

    # 5) combine
    top_alias_hits = _aggregate_candidates(alias_scores, syn_hits, topK=topK)
    
    # 6) fallback when no hits
    debug = {}
    if not top_alias_hits:
        inferred_intent = {
            "ask": "top_k",
            "metric": {"op": "sum" if _has_money_signal(tokens) else "count"},
            "target": _best_geo_field(iem),
        }
        debug["fallbackIntent"] = inferred_intent
        # create a minimal stub to still produce a vector
        stub = {
            "ask": "infer",
            "select": tokens if tokens else ["unknown"],
            "target": inferred_intent["target"]
        }
        vec_cosine = encode_intent_to_vocab(stub, iem.vocab)
        vec_bm25 = vec_cosine  # fallback: use same vec
        vec = _l2_normalize(vec_cosine)
    else:
        # 7) cosine side: small NL stub → use tokens + best targets as 'select'
        stub = {
            "ask": "infer",
            "select": tokens + [h["target"] for h in top_alias_hits[:3]],
            "target": top_alias_hits[0]["target"] if top_alias_hits else None
        }
        vec_cosine = encode_intent_to_vocab(stub, iem.vocab)

        # 8) BM25 vectorized from weighted targets
        vec_bm25 = _vectorize_targets_with_weights(top_alias_hits, iem.vocab)

        # 9) blend
        blended = (np.array(vec_cosine, dtype=float) * float(alpha_cosine)) + \
                  (np.array(vec_bm25, dtype=float) * float(1.0 - alpha_cosine))
        vec = _l2_normalize(blended.tolist())

    # 10) expose debug
    debug["corrections"] = corrections
    debug["fixedText"] = fixed_text

    return {
        "version": "intent-nl/0.1",
        "dim": len(iem.vocab),
        "vec": vec,
        "topAliasHits": top_alias_hits,
        "blend": {"cosine": float(alpha_cosine), "bm25": float(1.0 - alpha_cosine)},
        "debug": debug
    }