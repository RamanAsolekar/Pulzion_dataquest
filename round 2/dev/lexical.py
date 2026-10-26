"""Text normalisation + BM25 (numpy) — candidate code for solution."""
import math
import re

import numpy as np

try:
    import Stemmer  # PyStemmer

    _stem = Stemmer.Stemmer("english").stemWord
except Exception:  # pragma: no cover
    from nltk.stem.snowball import SnowballStemmer

    _stem = SnowballStemmer("english").stem

_STEM_CACHE: dict[str, str] = {}


def stem(w: str) -> str:
    s = _STEM_CACHE.get(w)
    if s is None:
        s = _stem(w) if w.isalpha() and len(w) > 3 else w
        _STEM_CACHE[w] = s
    return s


STOP = set(
    """a an the and or but if then else of to in on at by for with from into onto over under as is are was were be been being
    this that these those it its they them their there here which who whom whose what when where why how than so such
    do does did done doing has have had having can could may might must shall should will would not no nor also too very
    s t just only any all each both either neither more most other some same own about above below between through during
    before after again further once i me my we our you your he she his her us via per using used use based e g eg i e ie al et
    within without upon across against among around""".split()
)
# reviewer / question meta words: carry no topical information
QSTOP = set(
    """paper papers author authors study studies work manuscript article statement statements accurate accurately entirely
    really true correct correctly reasoning reason reasons behind justify justified justification explain explained explanation
    clarify clarified discuss discussed describe described mention mentioned provide provided give given state stated
    claim claimed exactly specifically particular particularly please could would should whether also specific
    mean meant means approach""".split()
)

_JUNK_RE = re.compile(r"\[\]|\[(?=\d)|(?<=\d)\]")
_PM_RE = re.compile(r"\b(PM|pm)\s?(\d{1,2})[\s.](\d)\b")
_DEC_FORMULA_RE = re.compile(r"\b([A-Za-z]{1,4}\d+)\.(\d+)\b")
_FORMULA_RE = re.compile(r"\b((?:(?:[A-Z][a-z]?|[A-Z]{2,4})\s{0,2}\d{1,2}\s{0,2})+)(?=\W|$)")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _join_formula(m: re.Match) -> str:
    s = m.group(1)
    # only join if there is at least one internal space (otherwise already joined)
    return s.replace(" ", "") + (" " if s.endswith(" ") else "")


def normalize(text: str) -> str:
    t = _JUNK_RE.sub(" ", text)
    t = t.replace("[", " ").replace("]", " ")
    t = _PM_RE.sub(r"\1\2\3", t)  # PM 2 5 / PM2.5 / PM 2.5 -> PM25
    t = _DEC_FORMULA_RE.sub(r"\1\2", t)
    t = _FORMULA_RE.sub(_join_formula, t)  # H 2 SO 4 -> H2SO4, CO 2 -> CO2
    return t


_CTOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def tokenize(text: str, query: bool = False) -> list[str]:
    out = []
    for raw in _CTOKEN_RE.findall(normalize(text)):
        tok = raw.lower()
        parts = [p for p in re.split(r"[-']", tok) if p]
        cands = ["".join(parts)] + parts if len(parts) > 1 else [tok]
        acro = raw.isupper() and len(raw) >= 2  # keep acronyms (AIRS, NOAA) unstemmed
        for w in cands:
            if w in STOP or (query and w in QSTOP):
                continue
            if len(w) == 1 and not w.isdigit():
                continue
            out.append(w if acro else stem(w))
    return out


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        dl = np.zeros(len(docs_tokens), dtype=np.float32)
        for i, toks in enumerate(docs_tokens):
            dl[i] = len(toks)
            tf: dict[int, int] = {}
            for w in toks:
                j = vocab.setdefault(w, len(vocab))
                tf[j] = tf.get(j, 0) + 1
            for j, c in tf.items():
                rows.append(i)
                cols.append(j)
                vals.append(c)
        self.vocab = vocab
        n = len(docs_tokens)
        V = len(vocab)
        tfm = np.zeros((V, n), dtype=np.float32)  # small corpora: dense term x doc
        tfm[cols, rows] = vals
        df = (tfm > 0).sum(1)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        avg = dl.mean() if n else 1.0
        norm = k1 * (1 - b + b * dl / avg)
        self.w = (tfm * (k1 + 1) / (tfm + norm[None, :])) * self.idf[:, None]

    def scores(self, q_tokens: list[str]) -> np.ndarray:
        idx = [self.vocab[w] for w in q_tokens if w in self.vocab]
        if not idx:
            return np.zeros(self.w.shape[1], dtype=np.float32)
        return self.w[idx].sum(0)
