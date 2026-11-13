"""Build data/pool/problems.jsonl: integer-answer olympiad problems + official train problems."""
import glob, hashlib, json, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows, seen = [], set()

def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()

def int_answer(a):
    s = str(a).strip().strip("$").replace(",", "").replace(r"\!", "")
    return str(int(s)) if re.fullmatch(r"-?\d+", s) else None

for f in sorted(glob.glob(str(ROOT / "data/pool/raw/*.parquet"))):
    src = Path(f).name.split("__")[0]
    df = pd.read_parquet(f)
    for _, r in df.iterrows():
        ans = int_answer(r["answer"])
        if ans is None or norm(r["problem"]) in seen:
            continue
        seen.add(norm(r["problem"]))
        rows.append({"pid": hashlib.sha1(norm(r["problem"]).encode()).hexdigest()[:12], "source": src,
                     "problem": r["problem"], "answer": ans})

# Official train problems (answers unknown -> None; solve rate measured by majority later)
tr = pd.read_parquet(ROOT / "data/raw/train-main-v2/data.parquet")
for p in tr.problem.unique():
    key = norm(p)
    hit = [x for x in rows if norm(x["problem"]) == key]
    if hit:
        hit[0]["in_train"] = True
        continue
    rows.append({"pid": hashlib.sha1(key.encode()).hexdigest()[:12], "source": "train-main-v2",
                 "problem": p, "answer": None, "in_train": True})

out = ROOT / "data/pool/problems.jsonl"
with out.open("w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), pd.Series([r["source"] for r in rows]).value_counts().to_dict())
print("train problems with known answers:", sum(1 for r in rows if r.get("in_train") and r["answer"]))
