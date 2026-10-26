import json, os, sys, time, getpass
sys.path.insert(0, "/app"); os.chdir("/app")
t0 = time.time()
import solution
print("user", getpass.getuser(), "uid", os.getuid(), "cpus", len(os.sched_getaffinity(0)), "HF_HOME", os.environ.get("HF_HOME"))
print("models loaded: encoder", solution._ENC is not None, "reranker", solution._RR is not None, "import_s", round(time.time() - t0, 1))
for l in open("/data/corpus.jsonl"):
    d = json.loads(l); solution.ingest(d["document_id"], d["document_text"])
out = [solution.retrieve(json.loads(l)["query"]) for l in open("/data/queries.jsonl")]
print("answers", len(out), "all valid", all(o for o in out))
print("RESULT", json.dumps(out))
try:
    open("/app/x.tmp", "w"); print("root writable?! BAD")
except OSError:
    print("root fs read-only: OK")
