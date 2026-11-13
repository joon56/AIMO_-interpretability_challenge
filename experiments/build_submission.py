"""Build submission ZIPs: solution.py + aimo_robust package + artifacts (+ small.txt for the small track).

Usage: python experiments/build_submission.py --art runs/artifacts --name v1
Writes submissions/<name>-main.zip and submissions/<name>-small.zip and appends SHA256 to submissions/MANIFEST.md.
"""

import argparse
import hashlib
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMALL = {"Qwen_Qwen3.5-4B", "Skywork_Skywork-OR1-Math-7B", "allenai_Olmo-3-7B-Think",
         "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"}

ap = argparse.ArgumentParser()
ap.add_argument("--art", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--note", default="")
args = ap.parse_args()

pkg = ROOT / "src/aimo_robust"
arts = sorted(Path(args.art).glob("*.npz"))
assert arts, "no artifacts"
(ROOT / "submissions").mkdir(exist_ok=True)
for track in ("main", "small"):
    out = ROOT / f"submissions/{args.name}-{track}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "src/solution.py", "solution.py")
        for f in sorted(pkg.glob("*.py")):
            z.write(f, f"aimo_robust/{f.name}")
        for a in arts:
            if track == "small" and a.stem.split("__")[0] not in SMALL:
                continue
            z.write(a, f"aimo_robust/artifacts/{a.name}")
        if track == "small":
            z.writestr("small.txt", "")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    with open(ROOT / "submissions/MANIFEST.md", "a", encoding="utf-8") as fh:
        fh.write(f"| {time.strftime('%Y-%m-%d %H:%M')} | {out.name} | {sha[:16]} | {out.stat().st_size} | {args.note} |\n")
    print(out.name, sha[:16], out.stat().st_size, [n for n in zipfile.ZipFile(out).namelist()])
