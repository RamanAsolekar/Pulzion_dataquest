"""DataQuest 2026 Senior R2 — hybrid retrieval with a gated cross-encoder.

Pipeline
  ingest:   normalise PDF noise -> BM25 tokens; dense embedding (arctic-embed-m-v1.5,
            fp32 ONNX) computed in micro-batches while documents arrive.
  retrieve: hybrid first stage = z(dense cosine) + 0.5 * z(BM25) over all docs.
            If the first stage is not clearly confident, the top-K candidates are
            rescored by a cross-encoder (gte-reranker-modernbert-base, fp32 ONNX)
            and fused with the first-stage score.
  safety:   time guard shrinks/disables reranking near the budget; any failure
            falls back to BM25 / first ingested id, so a valid id is always returned.
"""

import math
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

_T0 = time.time()

# ----------------------------------------------------------------------------- config
# fp32 ONNX is ~1.5 Hit@1 points better than the int8 exports on our dev sets; int8 kept as fallback
EMB_NAME, EMB_FILES = "snowflake-arctic-embed-m-v1.5", ("onnx/model.onnx", "onnx/model_int8.onnx")
RR_NAME, RR_FILES = "gte-reranker-modernbert-base", ("onnx/model.onnx", "onnx/model_int8.onnx")
Q_PREFIX = "Represent this sentence for searching relevant passages: "
BM25_K1, BM25_B = 1.2, 0.75
BM25_W = 0.5  # weight of z(BM25) relative to z(dense) in the first stage
RR_K = 5  # candidates rescored by the cross-encoder
RR_ALPHA = 0.75  # weight of z(first stage) when fused with z(cross-encoder)
GAP_SKIP = 2.0  # skip reranking when first-stage top1 - top2 exceeds this
EMB_BATCH = 8
MAX_LEN = 512
SOFT_S = 22 * 60  # beyond this: smaller K
HARD_S = 30 * 60  # beyond this: no reranking
EMB_STOP_S = 18 * 60  # stop embedding new documents during ingest (dense then disabled)

_HERE = Path(__file__).resolve().parent
_MODEL_ROOTS = [
    Path(p)
    for p in (
        os.environ.get("DQ_MODEL_DIR", ""),
        str(_HERE / "assets" / "models"),
        str(Path(os.environ.get("HF_HOME", "/opt/cache")) / "models"),
        "/opt/cache/models",
    )
    if p
]


def _n_threads() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 4
    return max(1, min(4, n))


# ----------------------------------------------------------------------------- text
try:
    import Stemmer  # PyStemmer

    _stem_fn = Stemmer.Stemmer("english").stemWord
except Exception:  # pragma: no cover - stemming is optional
    _stem_fn = None

_STEM_CACHE: dict = {}


def _stem(w: str) -> str:
    s = _STEM_CACHE.get(w)
    if s is None:
        s = _stem_fn(w) if (_stem_fn and w.isalpha() and len(w) > 3) else w
        _STEM_CACHE[w] = s
    return s


_STOP = set(
    """a an the and or but if then else of to in on at by for with from into onto over under as is are was were be been
    being this that these those it its they them their there here which who whom whose what when where why how than so
    such do does did done doing has have had having can could may might must shall should will would not no nor also
    too very s t just only any all each both either neither more most other some same own about above below between
    through during before after again further once i me my we our you your he she his her us via per using used use
    based e g eg i e ie al et within without upon across against among around""".split()
)
_QSTOP = set(
    """paper papers author authors study studies work manuscript article statement statements accurate accurately
    entirely really true correct correctly reasoning reason reasons behind justify justified justification explain
    explained explanation clarify clarified discuss discussed describe described mention mentioned provide provided
    give given state stated claim claimed exactly specifically particular particularly please whether specific mean
    meant means approach""".split()
)

_JUNK_RE = re.compile(r"\[\]|\[(?=\d)|(?<=\d)\]")
_PM_RE = re.compile(r"\b(PM|pm)\s?(\d{1,2})[\s.](\d)\b")
_DEC_FORMULA_RE = re.compile(r"\b([A-Za-z]{1,4}\d+)\.(\d+)\b")
_FORMULA_RE = re.compile(r"\b((?:(?:[A-Z][a-z]?|[A-Z]{2,4})\s{0,2}\d{1,2}\s{0,2})+)(?=\W|$)")
_CTOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_WS_RE = re.compile(r"\s+")


def _join_formula(m: re.Match) -> str:
    s = m.group(1)
    return s.replace(" ", "") + (" " if s.endswith(" ") else "")


def _normalize(text: str) -> str:
    t = _JUNK_RE.sub(" ", text).replace("[", " ").replace("]", " ")
    t = _PM_RE.sub(r"\1\2\3", t)  # PM 2 5 / PM2.5 / PM 2.5 -> PM25
    t = _DEC_FORMULA_RE.sub(r"\1\2", t)
    return _FORMULA_RE.sub(_join_formula, t)  # H 2 SO 4 -> H2SO4


_PM_MODEL_RE = re.compile(r"\b(PM|pm)\s(\d{1,2})\s(\d)\b")


def _model_text(text: str) -> str:
    """Clean-up for neural models: drop extraction junk, repair split formulas
    (PM 2 5 -> PM2.5, H 2 SO 4 -> H2SO4, CO 2 -> CO2), keep wording."""
    t = _JUNK_RE.sub(" ", text).replace("[", " ").replace("]", " ")
    t = _PM_MODEL_RE.sub(r"\1\2.\3", t)
    t = _FORMULA_RE.sub(_join_formula, t)
    return _WS_RE.sub(" ", t).strip()


def _tokenize(text: str, query: bool = False) -> list:
    out = []
    for raw in _CTOKEN_RE.findall(_normalize(text)):
        tok = raw.lower()
        parts = [p for p in re.split(r"[-']", tok) if p]
        cands = ["".join(parts)] + parts if len(parts) > 1 else [tok]
        acro = raw.isupper() and len(raw) >= 2  # keep acronyms (AIRS, NOAA) unstemmed
        for w in cands:
            if w in _STOP or (query and w in _QSTOP):
                continue
            if len(w) == 1 and not w.isdigit():
                continue
            out.append(w if acro else _stem(w))
    return out


# ----------------------------------------------------------------------------- BM25
class _BM25:
    """Okapi BM25 over sparse postings (memory ~ number of tokens)."""

    def __init__(self, docs_tokens: list, k1: float = BM25_K1, b: float = BM25_B):
        n = len(docs_tokens)
        self.n = n
        dl = np.array([len(t) for t in docs_tokens], dtype=np.float32)
        avg = float(dl.mean()) if n and dl.mean() > 0 else 1.0
        norm = k1 * (1.0 - b + b * dl / avg)
        post: dict = {}
        for i, toks in enumerate(docs_tokens):
            tf: dict = {}
            for w in toks:
                tf[w] = tf.get(w, 0) + 1
            for w, c in tf.items():
                post.setdefault(w, ([], []))
                post[w][0].append(i)
                post[w][1].append(c)
        self.post = {}
        for w, (docs, tfs) in post.items():
            d = np.array(docs, dtype=np.int32)
            tf = np.array(tfs, dtype=np.float32)
            idf = math.log(1.0 + (n - len(d) + 0.5) / (len(d) + 0.5))
            self.post[w] = (d, (tf * (k1 + 1.0) / (tf + norm[d]) * idf).astype(np.float32))

    def scores(self, q_tokens: list) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for w in q_tokens:  # repeated query terms count again (query tf)
            p = self.post.get(w)
            if p is not None:
                s[p[0]] += p[1]
        return s


# ----------------------------------------------------------------------------- ONNX models
def _find_model(name: str, fnames: tuple):
    for fname in fnames:
        for root in _MODEL_ROOTS:
            d = root / name
            if (d / fname).is_file() and (d / "tokenizer.json").is_file():
                return d, fname
    return None, None


def _session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = _n_threads()
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    so.enable_cpu_mem_arena = False  # keep peak RSS low; buffers are freed after each run
    so.enable_mem_pattern = False
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class _Encoder:
    """Bi-encoder with CLS pooling (arctic-embed / bge style)."""

    def __init__(self, d: Path, fname: str):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)
        self.tok.enable_padding()
        self.sess = _session(d / fname)
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts: list) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        h = self.sess.run(None, feed)[0]
        e = h[:, 0].astype(np.float32) if h.ndim == 3 else h.astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


class _CrossEncoder:
    def __init__(self, d: Path, fname: str):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN, strategy="only_second")
        self.tok.enable_padding()
        self.sess = _session(d / fname)
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def score(self, query: str, docs: list) -> np.ndarray:
        enc = self.tok.encode_batch([(query, d) for d in docs])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        lg = self.sess.run(None, feed)[0]
        return (lg[:, 0] if lg.ndim == 2 else lg).astype(np.float32)


_ENC = None
_RR = None
try:
    _d, _f = _find_model(EMB_NAME, EMB_FILES)
    if _d is not None:
        _ENC = _Encoder(_d, _f)
except Exception:  # pragma: no cover
    _ENC = None
try:
    _d, _f = _find_model(RR_NAME, RR_FILES)
    if _d is not None:
        _RR = _CrossEncoder(_d, _f)
except Exception:  # pragma: no cover
    _RR = None

# ----------------------------------------------------------------------------- state
_LOCK = threading.Lock()
_IDS: list = []
_TEXTS: list = []  # model-cleaned text
_TOKS: list = []
_EMB: list = []  # list of np arrays (batches)
_PENDING: list = []  # indices awaiting embedding
_bm25 = None
_D = None  # dense matrix N x dim
_built_n = -1


def _flush_embeddings() -> None:
    global _ENC
    if _ENC is None or not _PENDING:
        _PENDING.clear()
        return
    if time.time() - _T0 > EMB_STOP_S:  # corpus far larger than expected: lexical only
        _ENC = None
        _EMB.clear()
        _PENDING.clear()
        return
    try:
        _EMB.append(_ENC.encode([_TEXTS[i] for i in _PENDING]))
    except Exception:  # embedding failure -> lexical only
        _ENC = None
        _EMB.clear()
    _PENDING.clear()


def ingest(document_id: str, document_text: str) -> None:
    """Store one document. Called sequentially, once per document."""
    with _LOCK:
        text = document_text if isinstance(document_text, str) else str(document_text or "")
        _IDS.append(str(document_id))
        _TEXTS.append(_model_text(text))
        _TOKS.append(_tokenize(text))
        _PENDING.append(len(_IDS) - 1)
        if len(_PENDING) >= EMB_BATCH:
            _flush_embeddings()


def _ensure_index() -> None:
    global _bm25, _D, _built_n
    if _built_n == len(_IDS):
        return
    _flush_embeddings()
    _bm25 = _BM25(_TOKS)
    _D = None
    if _ENC is not None and _EMB:
        D = np.concatenate(_EMB, axis=0)
        if D.shape[0] == len(_IDS):
            _D = D
    _built_n = len(_IDS)


def _z(x: np.ndarray) -> np.ndarray:
    sd = float(x.std())
    return (x - float(x.mean())) / (sd if sd > 1e-9 else 1.0)


def _retrieve(query: str) -> str:
    _ensure_index()
    n = len(_IDS)
    lex = _bm25.scores(_tokenize(query, query=True))
    first = BM25_W * _z(lex)
    if _D is not None:
        try:
            q = _ENC.encode([Q_PREFIX + _model_text(query)])[0]
            first = first + _z(_D @ q)
        except Exception:
            pass
    order = np.argsort(-first, kind="stable")
    best = int(order[0])
    if _RR is None or n < 2:
        return _IDS[best]
    elapsed = time.time() - _T0
    if elapsed > HARD_S:
        return _IDS[best]
    k = min(RR_K if elapsed < SOFT_S else 3, n)
    if first[order[0]] - first[order[1]] > GAP_SKIP:
        return _IDS[best]
    cand = order[:k]
    try:
        ce = _RR.score(_model_text(query), [_TEXTS[i] for i in cand])
    except Exception:
        return _IDS[best]
    fused = _z(ce) + RR_ALPHA * _z(first[cand])
    return _IDS[int(cand[int(np.argmax(fused))])]


def retrieve(query: str) -> str:
    """Return exactly one document_id: the best match for this query."""
    with _LOCK:
        if not _IDS:
            return ""
        try:
            return _retrieve(query if isinstance(query, str) else str(query or ""))
        except Exception:
            try:
                s = _bm25.scores(_tokenize(str(query), query=True)) if _bm25 is not None else None
                if s is not None and len(s) == len(_IDS):
                    return _IDS[int(np.argmax(s))]
            except Exception:
                pass
            return _IDS[0]
