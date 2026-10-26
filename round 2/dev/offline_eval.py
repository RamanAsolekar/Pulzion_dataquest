"""Run solution.py in-process on a dev set (fresh process), 4 pinned cores.

usage: python offline_eval.py <dataset> [solution_dir]
"""
import importlib.util
import json
import os
import sys
import time

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "4"
os.environ.setdefault("DQ_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_models"))
import psutil

try:
    psutil.Process().cpu_affinity([0, 2, 4, 6])
except Exception:
    pass
DEV = os.path.dirname(os.path.abspath(__file__))
ds = sys.argv[1]
sol_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DEV, "..", "dataquest-2026-senior-R2")
d = os.path.join(DEV, "data", ds) if not os.path.isdir(ds) else ds
corpus = [json.loads(l) for l in open(os.path.join(d, "corpus.jsonl"), encoding="utf-8") if l.strip()]
queries = [json.loads(l) for l in open(os.path.join(d, "queries.jsonl"), encoding="utf-8") if l.strip()]
qrels = json.load(open(os.path.join(d, "qrels.json")))

t0 = time.time()
spec = importlib.util.spec_from_file_location("solution", os.path.join(sol_dir, "solution.py"))
sol = importlib.util.module_from_spec(spec)
sys.path.insert(0, sol_dir)
spec.loader.exec_module(sol)
t1 = time.time()
for c in corpus:
    sol.ingest(c["document_id"], c["document_text"])
t2 = time.time()
hits, out = 0, []
for q in queries:
    r = sol.retrieve(q["query"])
    h = r in qrels.get(q["query_id"], [])
    hits += h
    out.append({"query_id": q["query_id"], "query": q["query"], "returned": r, "relevant": qrels.get(q["query_id"]), "hit": int(h)})
t3 = time.time()
rss = psutil.Process().memory_info().peak_wset / 2**20 if hasattr(psutil.Process().memory_info(), "peak_wset") else psutil.Process().memory_info().rss / 2**20
print(json.dumps({"dataset": ds, "n_docs": len(corpus), "n_queries": len(queries), "hit1": round(hits / len(queries), 4),
                  "import_s": round(t1 - t0, 1), "ingest_s": round(t2 - t1, 1), "retrieve_s": round(t3 - t2, 1),
                  "total_s": round(t3 - t0, 1), "peak_mem_mib": round(rss), "enc": sol._ENC is not None, "rr": sol._RR is not None}))
json.dump(out, open(os.path.join(DEV, "cache", f"offline_{ds}.json"), "w"), indent=1)
