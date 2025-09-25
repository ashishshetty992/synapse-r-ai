import math
from typing import Dict, Any, List, Tuple
from collections import defaultdict

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