"""Build a DataQuest R2 submission ZIP containing only participant-owned files.

Whitelist:
    solution.py              required
    requirements.txt         optional
    prefetch.py              optional
    assets/**                optional
    extra root-level *.py    optional, except harness-owned/template-only names

Everything else is ignored. The harness supplies server.py, Dockerfile,
.dockerignore, and requirements.server.txt.

Usage:
    python make_submission.py --output submission.zip
"""

import argparse
import sys
import zipfile
from pathlib import Path

MAX_ZIP_BYTES = 512 * 1024**2
ROOT = Path(__file__).resolve().parent

CORE_FILES = ("solution.py", "requirements.txt", "prefetch.py")
HARNESS_OWNED = ("server.py", "Dockerfile", ".dockerignore", "requirements.server.txt")
TEMPLATE_ONLY = (
    "README.md",
    "local_eval.py",
    "make_submission.py",
    "eval_log.json",
)
EXCLUDED_DIRS = {".git", ".venv", "__pycache__", "sample_data", "assets/__pycache__"}
EXCLUDED_NAMES = {".gitkeep", ".DS_Store"}


def should_include_root_py(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix == ".py"
        and path.name not in CORE_FILES
        and path.name not in HARNESS_OWNED
        and path.name not in TEMPLATE_ONLY
    )


def iter_submission_files(root: Path):
    solution = root / "solution.py"
    if not solution.is_file():
        raise FileNotFoundError(f"missing required file: {solution}")

    yield solution
    for name in ("requirements.txt", "prefetch.py"):
        p = root / name
        if p.is_file():
            yield p

    assets = root / "assets"
    if assets.is_dir():
        for p in sorted(assets.rglob("*")):
            if (
                p.is_file()
                and not p.is_symlink()
                and p.name not in EXCLUDED_NAMES
                and not any(part in EXCLUDED_DIRS for part in p.parts)
            ):
                yield p

    for p in sorted(root.glob("*.py")):
        if should_include_root_py(p):
            yield p


def build_zip(root: Path, output: Path) -> None:
    files = list(iter_submission_files(root))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(root).as_posix())

    size = output.stat().st_size
    print(f"Wrote {output} ({size} bytes, {len(files)} files)")
    for p in files:
        print(f"  {p.relative_to(root).as_posix()}")
    if size > MAX_ZIP_BYTES:
        print(
            f"ERROR: ZIP is larger than the 512 MB limit ({size} > {MAX_ZIP_BYTES} bytes)",
            file=sys.stderr,
        )
        output.unlink(missing_ok=True)
        raise SystemExit(1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create a whitelisted DataQuest submission ZIP.")
    parser.add_argument(
        "--output",
        default=str(ROOT / "submission.zip"),
        help="output ZIP path (default: ./submission.zip)",
    )
    parser.add_argument(
        "--root", default=str(ROOT), help="template root (default: script directory)"
    )
    args = parser.parse_args(argv)
    build_zip(Path(args.root).resolve(), Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
