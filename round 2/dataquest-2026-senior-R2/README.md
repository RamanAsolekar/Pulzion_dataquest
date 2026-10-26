# DataQuest Round 2 (RAG) — Participant Template

Minimal template: implement two functions, everything else is harness-owned.

## Contract

```python
def ingest(document_id: str, document_text: str) -> None
def retrieve(query: str) -> str  # exactly one document_id
```

- `ingest` is called once per document, sequentially. The query set is unknown.
- `retrieve` is called once per query, sequentially, only after the last `ingest`.
- No third lifecycle function. Heavy index building (embeddings, ANN, BM25 fit)
  may be deferred to the first `retrieve` call; that time counts toward the
  40-minute total execution budget.
- State lives in process memory or under `/tmp`. The root filesystem is
  read-only at official evaluation time. One process, one worker.
- At evaluation time files are copied into `/app`: `solution.py` at
  `/app/solution.py`, extra root modules alongside it, and `assets/**` at
  `/app/assets/...`. The process runs as `runner` (UID 1000) with working
  directory `/app`. Use `Path(__file__).resolve().parent / "assets"` to find
  assets; write runtime state only under `/tmp`.
- `solution.py` ships with a working BM25 baseline — replace it freely.

Return exactly one document ID. Returning `""`, a list, an unknown ID, or
raising an error scores 0 for that question. There is no `k` and no ranked list.

## Files you may edit

- `solution.py` (required), `requirements.txt` (optional; the shipped BM25
  baseline needs `rank-bm25`, so keep this file if you keep the baseline),
  `prefetch.py` (optional), `assets/**` (optional), and extra root-level
  `.py` modules. The template includes an empty `assets/` directory; put
  model weights, indexes, or data files there.
- Do **not** rely on changes to `server.py`, `Dockerfile`, `.dockerignore`, or
  `requirements.server.txt` — the harness overwrites those four with its own
  copies at official evaluation time.
- Do not submit `README.md`, `local_eval.py`, `make_submission.py`,
  `sample_data/`, `eval_log.json`, `.git/`, caches, or virtual environments.

`prefetch.py` runs at **build time** (normal network available) as root, before
the runtime user is switched — put all model/data downloads there. The image
sets `HF_HOME=/opt/cache` and keeps it set at runtime. Download Hugging Face
models into the normal cache under `/opt/cache`, or into an explicit directory
such as `/app/assets/models`, and load from that exact path at runtime. Do not
rely on `~/.cache`; `HOME=/tmp` at runtime. Do not use `/tmp` for build
artifacts; runtime `/tmp` is a tmpfs and hides build-time contents. At runtime
there is no network.

## Limits

- Build: **20 minutes**; image: **2.5 GB**.
- Execution: **40 minutes total** for startup + model loading + all ingest and
  retrieve calls.
- Memory: **2.5 GB**; writable disk: **1 GB** (`/tmp` tmpfs only); CPU only.
- Runtime network: **none**.
- Submission ZIP: **512 MB**.
- The container root filesystem is read-only. Use `/tmp` or memory only.
  Environment: `HOME=/tmp`, `TMPDIR=/tmp`, `XDG_CACHE_HOME=/tmp/.cache`,
  `PYTHONDONTWRITEBYTECODE=1`, `HF_HOME=/opt/cache`, `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`.
- `/tmp` is a tmpfs, so everything written there is charged to the same 2.5 GB
  memory limit. Filling `/tmp` leaves that much less for your Python process.

If the execution deadline is reached, completed questions are scored and the
rest score 0. A run is disqualified only if no question completed.

## Evaluation machine

Every submission runs on the same machine profile, and `local_eval.py`
reproduces it by default:

| | |
|---|---|
| CPU | AMD Ryzen 7 7840HS (Zen 4) |
| Cores | **4 physical cores, SMT disabled** (`--cpuset-cpus`) |
| Clock | **3.8 GHz fixed**; turbo/boost disabled; governor `performance` |
| SIMD | AVX2, AVX-512 (VNNI, BF16) |
| Threads | **4** (`OMP`/`MKL`/`OPENBLAS`/`NUMEXPR_NUM_THREADS=4`) |
| Python / BLAS | 3.11 / OpenBLAS (PyPI wheel) |

`local_eval.py --cpuset auto` picks one thread per physical core on your machine
and applies the same thread caps, so local timings match the official limits.
Inside the container `os.cpu_count()` may still report the host's full core
count; use `len(os.sched_getaffinity(0))`, which `GET /health` returns as
`cpus`.

## Metric

**Hit@1**: return exactly one document ID per question. Score 1 if that ID is
one of the relevant documents for the question, otherwise 0. If a question has
multiple relevant documents, returning any one of them scores 1. Final score is
the mean over all questions (0–1).

This is **not Recall@k** and **not Recall@1**. There is no top-k truncation,
deduplication, or backfilling.

Tiebreakers: lower total execution time (startup + ingest + retrievals), lower
peak memory, earlier submission.

## Local eval

```bash
pip install requests
python local_eval.py                 # sample data, official isolation defaults
python local_eval.py --no-build --tag dataquest-local
python local_eval.py --help          # all options
```

Builds the image, pins it to 4 physical cores with a read-only root and 1 GB
`/tmp`, POSTs the sample docs then sample queries, prints Hit@1 + timings + peak
memory, and
writes `eval_log.json` (overwritten each run; `--log-file` to change). Ingest
errors are logged and skipped; retrieve errors score 0 for that question.
Container is stopped/removed unless `--keep-container` is passed.

Local runs may have internet access. The official run does not. The Dockerfile
sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` after `prefetch.py`, so
accidental runtime downloads fail locally too.

## Submission

Build the ZIP with the whitelist enforced:

```bash
python make_submission.py --output submission.zip
```

The ZIP should contain `solution.py` plus any optional participant-owned files.
The harness supplies `server.py`, `Dockerfile`, `.dockerignore`, and
`requirements.server.txt`; do not include them. See the competition README
Section 7 for the full list.
