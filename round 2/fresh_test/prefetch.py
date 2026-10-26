"""Build-time download of the ONNX models (pinned revisions, verified sizes).

Order: small int8 exports first (always fit in the build window), then the fp32
exports that solution.py prefers (~1.5 Hit@1 points better). A download that
would push the build past FP32_DEADLINE_S is abandoned; solution.py then uses
the int8 files. Files land in $HF_HOME/models/<name>/ (HF_HOME=/opt/cache).
Standard library only.
"""

import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

T0 = time.time()
FP32_DEADLINE_S = 12 * 60  # leave margin inside the 20-minute build limit
HOSTS = ["https://huggingface.co", "https://hf-mirror.com"]
HERE = Path(__file__).resolve().parent
DEST = Path(os.environ.get("HF_HOME", "/opt/cache")) / "models"

EMB = ("snowflake-arctic-embed-m-v1.5", "Snowflake/snowflake-arctic-embed-m-v1.5", "e58a8f756156a1293d763f17e3aae643474e9b8a")
RR = ("gte-reranker-modernbert-base", "Alibaba-NLP/gte-reranker-modernbert-base", "f7481e6055501a30fb19d090657df9ec1f79ab2c")
# (model, path, expected size, hard deadline or None)
PLAN = [
    (EMB, "tokenizer.json", 711649, None),
    (RR, "tokenizer.json", 3583499, None),
    (EMB, "onnx/model_int8.onnx", 110145162, None),
    (RR, "onnx/model_int8.onnx", 150871837, None),
    (RR, "onnx/model.onnx", 598803940, FP32_DEADLINE_S),
    (EMB, "onnx/model.onnx", 435939137, FP32_DEADLINE_S),
]


class _Late(Exception):
    pass


def _fetch(url: str, out: Path, deadline) -> None:
    tmp = out.with_name(out.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dataquest-prefetch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                if deadline is not None and time.time() - T0 > deadline:
                    raise _Late()
        tmp.replace(out)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> None:
    for (name, repo, rev), path, size, deadline in PLAN:
        if (HERE / "assets" / "models" / name / path).is_file():
            continue
        out = DEST / name / path
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.is_file() and out.stat().st_size == size:
            continue
        for attempt in range(6):
            if deadline is not None and time.time() - T0 > deadline:
                print(f"[prefetch] skip {name}/{path}: build-time budget used", file=sys.stderr)
                break
            host = HOSTS[attempt % len(HOSTS)]
            url = f"{host}/{repo}/resolve/{rev}/{path}"
            try:
                t = time.time()
                _fetch(url, out, deadline)
                if out.stat().st_size != size:
                    out.unlink()
                    raise IOError("size mismatch")
                print(f"[prefetch] {name}/{path} ({size >> 20} MiB) in {time.time() - t:.0f}s from {host}")
                break
            except _Late:
                print(f"[prefetch] abandoned {name}/{path}: build-time budget used", file=sys.stderr)
                break
            except Exception as e:  # retry with backoff / other host
                print(f"[prefetch] {url} failed: {e!r}", file=sys.stderr)
                time.sleep(3 * (attempt + 1))
    print(f"[prefetch] done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
