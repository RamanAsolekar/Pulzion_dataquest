"""Local evaluation harness (host-only dependency: `requests`).

Builds the repo image, runs it with the official isolation defaults, drives
ingest/retrieve over HTTP, and computes Hit@1:

    - `retrieve(query)` must return exactly one document ID
    - score 1 if that ID is one of the relevant documents, else 0
    - if several documents are relevant, any one of them scores 1
    - final score = mean over queries

Failure / timeout policy (matches the official harness):
    - an error on one /ingest call is logged; ingestion continues
    - an error on one /retrieve call scores 0 for that query
    - when the execution deadline is reached, completed queries are scored and
      the remaining queries score 0; a run with zero completed queries is
      reported as disqualified

Writes eval_log.json (overwritten each run; see --log-file) and prints a human
summary. Always stops/removes the container unless --keep-container is passed.

Example:
    pip install requests
    python local_eval.py
    python local_eval.py --no-build --tag dataquest-local --port 8080
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _read_text(path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _parse_cpu_list(spec: str) -> list[int]:
    cpus: set[int] = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def derive_cpuset(cores: int = 4) -> str | None:
    """One thread from each of `cores` physical cores, SMT siblings left idle.

    Mirrors the official runtime pinning. Linux only; returns None elsewhere.
    """
    sys_cpu = Path("/sys/devices/system/cpu")
    try:
        online = _parse_cpu_list(_read_text(sys_cpu / "online") or "0")
    except ValueError:
        return None
    seen: set[int] = set()
    picked: list[int] = []
    for cpu in online:
        siblings = _read_text(sys_cpu / f"cpu{cpu}" / "topology" / "thread_siblings_list")
        if not siblings:
            return None
        try:
            key = min(_parse_cpu_list(siblings))
        except ValueError:
            return None
        if key in seen:
            continue
        seen.add(key)
        picked.append(cpu)
        if len(picked) == cores:
            break
    if len(picked) < cores:
        return None
    return ",".join(str(cpu) for cpu in picked)


class PeakMemorySampler:
    """Sample `docker stats` while the container runs."""

    def __init__(self, container_name: str, interval_s: float = 0.5):
        self.container_name = container_name
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=5)
        return self.peak_bytes

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "docker",
                        "stats",
                        "--no-stream",
                        "--format",
                        "{{.MemUsage}}",
                        self.container_name,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if out.returncode == 0 and out.stdout.strip():
                    part = out.stdout.strip().split("/")[0].strip()
                    self.peak_bytes = max(self.peak_bytes, parse_docker_mem(part))
            except (subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_s)


def parse_docker_mem(s: str) -> int:
    s = s.strip()
    units = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    for suffix, mult in sorted(units.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(float(s))


def hit_at_1(document_id, relevant, corpus_ids) -> bool:
    return (
        isinstance(document_id, str) and document_id in corpus_ids and document_id in set(relevant)
    )


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(cmd, **kwargs):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def wait_for_health(base_url, deadline):
    while True:
        if time.time() >= deadline:
            raise TimeoutError("container did not become healthy before the execution deadline")
        try:
            r = requests.get(base_url + "/health", timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)


def request_timeout(deadline, max_s):
    remaining = deadline - time.time()
    if remaining <= 0:
        return 0.0
    return max(0.1, min(max_s, remaining))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Local eval for DataQuest R2 template.")
    p.add_argument("--tag", default="dataquest-local", help="docker image tag")
    p.add_argument("--port", type=int, default=8080, help="host port (bound on 127.0.0.1)")
    p.add_argument("--corpus", default=str(BASE_DIR / "sample_data" / "corpus.jsonl"))
    p.add_argument("--queries", default=str(BASE_DIR / "sample_data" / "queries.jsonl"))
    p.add_argument("--qrels", default=str(BASE_DIR / "sample_data" / "qrels.json"))
    p.add_argument(
        "--log-file",
        default=str(BASE_DIR / "eval_log.json"),
        help="eval log path (overwritten each run)",
    )
    p.add_argument("--build-timeout", type=float, default=1200.0)
    p.add_argument("--execution-timeout", type=float, default=2400.0)
    p.add_argument("--request-timeout", type=float, default=120.0)
    p.add_argument(
        "--keep-container", action="store_true", help="do not stop/remove the container after eval"
    )
    p.add_argument("--no-build", action="store_true", help="skip docker build, reuse tag")
    # Official isolation defaults. `--writable` is for local debugging only.
    p.add_argument("--memory", default="2.5g")
    p.add_argument("--memory-swap", default="2.5g")
    p.add_argument("--cpus", default=None, help="e.g. 4 ; unset by default")
    p.add_argument("--pids-limit", default="512")
    p.add_argument("--tmpfs-size", default="1g")
    p.add_argument(
        "--cpuset",
        default="auto",
        help=(
            'pin the container like the official run: "auto" picks one thread per '
            'physical core (SMT idle), "none" disables pinning, or pass a list '
            "such as 0,2,4,6"
        ),
    )
    p.add_argument(
        "--threads",
        type=int,
        default=4,
        help="value for OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS (default: 4)",
    )
    p.add_argument(
        "--writable",
        action="store_true",
        help="disable --read-only (debug only; official runs are read-only)",
    )
    p.add_argument("--http-proxy", default=None, help="pass --build-arg HTTP_PROXY=...")
    p.add_argument("--https-proxy", default=None, help="pass --build-arg HTTPS_PROXY=...")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    corpus = load_jsonl(args.corpus)
    queries = load_jsonl(args.queries)
    with open(args.qrels, encoding="utf-8") as f:
        qrels = json.load(f)
    corpus_ids = {d["document_id"] for d in corpus}

    if not args.no_build:
        build_cmd = ["docker", "build", "-t", args.tag, str(BASE_DIR)]
        for name, val in (("HTTP_PROXY", args.http_proxy), ("HTTPS_PROXY", args.https_proxy)):
            if val:
                build_cmd += ["--build-arg", f"{name}={val}"]
        try:
            run(build_cmd, timeout=args.build_timeout)
        except subprocess.TimeoutExpired:
            print(f"Build exceeded {args.build_timeout:.0f}s", file=sys.stderr)
            return 1

    cpuset = None
    if args.cpuset and args.cpuset.strip().lower() not in ("none", "off", ""):
        cpuset = (
            derive_cpuset(args.threads)
            if args.cpuset.strip().lower() == "auto"
            else args.cpuset.strip()
        )
        if not cpuset and args.cpuset.strip().lower() == "auto":
            print(
                "Warning: could not derive a physical-core cpuset on this host; "
                "running unpinned. Pass --cpuset none to silence this.",
                file=sys.stderr,
            )
    print(f"Runtime: cpuset={cpuset or 'off'} threads={args.threads}", flush=True)

    name = f"dataquest-eval-{uuid.uuid4().hex[:8]}"
    run_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "-p",
        f"127.0.0.1:{args.port}:8080",
        "--memory",
        args.memory,
        "--memory-swap",
        args.memory_swap,
        "--cap-drop=ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(args.pids_limit),
        "-e",
        "HOME=/tmp",
        "-e",
        "TMPDIR=/tmp",
        "-e",
        "XDG_CACHE_HOME=/tmp/.cache",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
    ]
    if cpuset:
        run_cmd += ["--cpuset-cpus", cpuset]
    for key in THREAD_ENV_KEYS:
        run_cmd += ["-e", f"{key}={args.threads}"]
    if args.cpus:
        run_cmd += ["--cpus", str(args.cpus)]
    if not args.writable:
        run_cmd += ["--read-only", "--tmpfs", f"/tmp:rw,size={args.tmpfs_size},mode=1777"]
    run_cmd.append(args.tag)

    base_url = f"http://127.0.0.1:{args.port}"
    container_started = time.time()
    run(run_cmd)
    sampler = PeakMemorySampler(name)
    sampler.start()

    ingest_errors, retrieve_errors, per_query = [], [], []
    startup_time = ingest_time = retrieval_time = 0.0
    timed_out = False
    completed_queries = 0
    score = 0.0
    try:
        deadline = container_started + args.execution_timeout
        wait_for_health(base_url, deadline)
        startup_time = time.time() - container_started

        ingest_start = time.time()
        for d in corpus:
            if time.time() >= deadline:
                timed_out = True
                break
            try:
                r = requests.post(
                    base_url + "/ingest",
                    json={"document_id": d["document_id"], "document_text": d["document_text"]},
                    timeout=request_timeout(deadline, args.request_timeout),
                )
                r.raise_for_status()
            except Exception as e:  # failure policy: log and continue
                ingest_errors.append({"document_id": d["document_id"], "error": repr(e)})
                if time.time() >= deadline:
                    timed_out = True
                    break
        ingest_time = time.time() - ingest_start

        retrieval_start = time.time()
        hits = 0
        for q in queries:
            if time.time() >= deadline:
                timed_out = True
                per_query.append(
                    {
                        "query_id": q["query_id"],
                        "relevant": qrels.get(q["query_id"], []),
                        "returned": None,
                        "hit": 0,
                        "error": "not evaluated: execution deadline exceeded",
                    }
                )
                continue
            qid = q["query_id"]
            relevant = qrels.get(qid, [])
            try:
                r = requests.post(
                    base_url + "/retrieve",
                    json={"query": q["query"]},
                    timeout=request_timeout(deadline, args.request_timeout),
                )
                r.raise_for_status()
                document_id = r.json().get("document_id")
            except Exception as e:  # failure policy: 0 for this query
                retrieve_errors.append({"query_id": qid, "error": repr(e)})
                document_id = None
            hit = hit_at_1(document_id, relevant, corpus_ids)
            hits += int(hit)
            completed_queries += 1
            per_query.append(
                {
                    "query_id": qid,
                    "relevant": relevant,
                    "returned": document_id,
                    "hit": int(hit),
                    "error": next(
                        (x["error"] for x in retrieve_errors if x["query_id"] == qid), None
                    ),
                }
            )
        retrieval_time = time.time() - retrieval_start
        total_time = time.time() - container_started
        score = hits / len(queries) if queries else 0.0
    except Exception as e:
        print(f"Evaluation failed: {e!r}", file=sys.stderr)
        return 1
    finally:
        peak_memory_bytes = sampler.stop()
        if not args.keep_container:
            subprocess.run(["docker", "stop", name], capture_output=True)
            subprocess.run(["docker", "rm", name], capture_output=True)
        else:
            print(f"Keeping container {name}", flush=True)

    disqualified = completed_queries == 0
    log = {
        "config": {
            "tag": args.tag,
            "port": args.port,
            "memory": args.memory,
            "memory_swap": args.memory_swap,
            "cpus": args.cpus,
            "cpuset": cpuset,
            "threads": args.threads,
            "pids_limit": args.pids_limit,
            "read_only": not args.writable,
            "tmpfs_size": None if args.writable else args.tmpfs_size,
            "execution_timeout_s": args.execution_timeout,
            "n_docs": len(corpus),
            "n_queries": len(queries),
        },
        "startup_time_s": startup_time,
        "ingest_time_s": ingest_time,
        "retrieval_time_s": retrieval_time,
        "total_time_s": total_time,
        "peak_memory_bytes": peak_memory_bytes,
        "score": score,
        "metric": "Hit@1",
        "timed_out": timed_out,
        "completed_queries": completed_queries,
        "disqualified": disqualified,
        "per_query": per_query,
        "ingest_errors": ingest_errors,
        "retrieve_errors": retrieve_errors,
    }
    with open(args.log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"Hit@1: {score:.4f} (mean over {len(queries)} queries)")
    print(f"Startup: {startup_time:.2f}s")
    print(f"Ingest time: {ingest_time:.2f}s")
    print(f"Retrieval time: {retrieval_time:.2f}s")
    print(f"Total execution: {total_time:.2f}s")
    print(f"Peak memory: {peak_memory_bytes / (1024 * 1024):.1f} MiB")
    print(f"Completed queries: {completed_queries}/{len(queries)}")
    print(f"Ingest errors: {len(ingest_errors)}, retrieve errors: {len(retrieve_errors)}")
    print(f"Eval log: {args.log_file}")
    if disqualified:
        print(
            "DISQUALIFIED: no questions completed before the execution deadline.", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
