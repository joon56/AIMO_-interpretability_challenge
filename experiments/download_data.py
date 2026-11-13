"""Download the public datasets the pipeline needs. This repository ships no problem text, so run this first.

    python experiments/download_data.py            # official labelled train rows + the MathArena problem-pool sources
    python experiments/download_data.py --all      # also the other aimo-interp sample datasets (exploratory analysis only)
    python experiments/download_data.py --build    # ... then build data/pool/problems.jsonl and problems_olmo.jsonl

Written under the repository root (all git-ignored):
    data/raw/<name>/{README.md,data.parquet}                  aimo-interp/<name> on the Hugging Face Hub
    data/pool/raw/<name>__<file>.parquet                      MathArena/<name> (competition problems with integer answers)
    data/pool/problems.jsonl, data/pool/problems_olmo.jsonl   with --build (see experiments/build_pool.py)

Datasets keep their own licenses; see data/README.md. Only the standard library is needed (pandas for --build).
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

try:  # corporate/intercepting proxies: use the OS trust store when available
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
HUB = "https://huggingface.co"
AIMO_REQUIRED = ["train-main-v2"]
AIMO_OPTIONAL = ["val-sample", "aimo-interp-challenge-sample-full", "augmented-sample-math",
                 "augmented-sample-math-agg", "augmented-sample-math-agg-filtered",
                 "augmented-sample-math-full-filtered"]
MATHARENA = ["aime_2023_I", "aime_2023_II", "aime_2024_I", "aime_2024_II", "aime_2025", "aime_2026", "hmmt_feb_2023",
             "hmmt_feb_2024", "hmmt_feb_2025", "hmmt_feb_2026", "hmmt_nov_2025", "brumo_2025", "cmimc_2025", "smt_2025",
             "apex_2025", "apex-shortlist"]
# subset used for the slower models (Olmo-3 / DeepSeek): every official train problem + six recent contests
SUBSET_SOURCES = {"aime_2025", "aime_2026", "hmmt_feb_2025", "hmmt_feb_2026", "brumo_2025", "cmimc_2025"}


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())
    print(f"  {dest.relative_to(ROOT)}  ({dest.stat().st_size / 1024:.0f} KiB)")


def parquet_files(repo: str) -> list[str]:
    with urllib.request.urlopen(f"{HUB}/api/datasets/{repo}", timeout=60) as resp:
        siblings = json.load(resp).get("siblings", [])
    return [s["rfilename"] for s in siblings if s["rfilename"].endswith(".parquet")]


def get_aimo(name: str) -> None:
    print(f"aimo-interp/{name}")
    files = parquet_files(f"aimo-interp/{name}")
    fetch(f"{HUB}/datasets/aimo-interp/{name}/resolve/main/README.md", ROOT / "data/raw" / name / "README.md")
    fetch(f"{HUB}/datasets/aimo-interp/{name}/resolve/main/{files[0]}", ROOT / "data/raw" / name / "data.parquet")


def get_matharena(name: str) -> None:
    print(f"MathArena/{name}")
    for f in parquet_files(f"MathArena/{name}"):
        fetch(f"{HUB}/datasets/MathArena/{name}/resolve/main/{f}", ROOT / "data/pool/raw" / f"{name}__{Path(f).name}")


def build() -> None:
    subprocess.run([sys.executable, str(ROOT / "experiments/build_pool.py")], check=True)
    pool = [json.loads(line) for line in open(ROOT / "data/pool/problems.jsonl", encoding="utf-8")]
    subset = [p for p in pool if p.get("in_train") or p["source"] in SUBSET_SOURCES]
    with open(ROOT / "data/pool/problems_olmo.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in subset)
    print(f"problems_olmo.jsonl: {len(subset)} problems ({sum(1 for p in subset if p.get('in_train'))} official train)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="also fetch the optional aimo-interp datasets")
    ap.add_argument("--build", action="store_true", help="build data/pool/problems*.jsonl after downloading")
    args = ap.parse_args()
    for n in AIMO_REQUIRED + (AIMO_OPTIONAL if args.all else []):
        get_aimo(n)
    for n in MATHARENA:
        get_matharena(n)
    if args.build:
        build()
