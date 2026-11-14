"""Offline decision-path test of a submission ZIP: the zip's own predict.py + artifacts, fake model/tokenizer/sampler.

For every config in the zip: the artifact loads, the expected path runs (sampling + combined decision, or probe-only),
the [aimo] line is printed, and the returned bool equals the artifact's threshold rule on the printed feature.
Also checks the sampling-failure fallback. Needs torch (CPU is fine); no model weights.
Usage: python experiments/mock_predict_test.py submissions/v3-main.zip [more zips]
"""

import contextlib
import importlib
import io
import re
import sys
import tempfile
import zipfile

import numpy as np

CONFIGS = [("Qwen/Qwen3.5-4B", "default"), ("Skywork/Skywork-OR1-Math-7B", "default"),
           ("allenai/Olmo-3-7B-Think", "default"), ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default"),
           ("openai/gpt-oss-120b", "low"), ("openai/gpt-oss-120b", "medium"), ("openai/gpt-oss-120b", "high")]


def run_zip(path: str, trials: int = 12) -> bool:
    tmp = tempfile.mkdtemp()
    zipfile.ZipFile(path).extractall(tmp)
    for mod in [m for m in sys.modules if m.startswith("aimo_robust")]:
        del sys.modules[mod]
    sys.path.insert(0, tmp)
    try:
        P = importlib.import_module("aimo_robust.predict")
        S = importlib.import_module("aimo_robust.sampler")
    finally:
        sys.path.remove(tmp)
    rng = np.random.default_rng(0)
    P.load_tokenizer = lambda m: None
    P.load_model = lambda m: None

    def fake_states(model, tok, model_id, problem, effort, wants):
        art = P.load_artifact(model_id, effort)
        out = {}
        for suf in ("", "_len"):
            if f"w{suf}" in art and (key := (int(art[f"layers{suf}"][0]), str(art[f"pool{suf}"]))) in wants:
                out[key] = (art[f"mu{suf}"] + art[f"sd{suf}"] * rng.normal(size=art[f"mu{suf}"].shape)).astype(np.float32)
        return out

    def fake_sample(model, tok, model_id, problem, effort, k, max_new, t_stop, seed=0, force_until=0.0,
                    capture_layer=0):
        if fail["on"]:
            raise RuntimeError("simulated sampler failure")
        art = P.load_artifact(model_id, effort)
        calls.append((model_id, effort, k, max_new, capture_layer))
        res = []
        for i in range(k):
            done = rng.random() < 0.4
            r = {"finished": done, "len": int(rng.integers(200, max_new)) if done else max_new,
                 "ans": str(rng.integers(0, 3)) if done else None, "forced": None}
            if capture_layer:
                r["state"] = (art["prog_mu"] + art["prog_sd"] * rng.normal(size=art["prog_mu"].shape)).astype(np.float32)
            res.append(r)
        return res

    P.pooled_states = fake_states
    S.sample = fake_sample
    ok_all, fail = True, {"on": False}
    for m, e in CONFIGS:
        art = P.load_artifact(m, e)
        if art is None:
            continue
        calls = []
        n_true = mismatches = paths = 0
        for t in range(trials):
            fail["on"] = t == trials - 1  # last trial: sampler raises -> fallback path
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                pred = P.predict_one(m, "Find 1+1.", e)
            line = [l for l in buf.getvalue().splitlines() if l.startswith("[aimo]")][-1]
            kv = dict(x.split("=", 1) for x in line.split()[3:] if "=" in x)
            feats, logit = {k: float(v.rstrip("s")) for k, v in kv.items() if k not in ("logit", "t")}, float(kv["logit"])
            use_comb = "comb_w" in art and m in P.SAMPLING and all(str(n) in feats for n in art["comb_feats"])
            names, w, b = ((art["comb_feats"], art["comb_w"], art["comb_b"]) if use_comb
                           else (art["cal_feats"], art["cal_w"], art["cal_b"]))
            expect = float(b) + sum(float(wi) * feats[str(n)] for n, wi in zip(names, w))
            if not (pred == (logit >= 0) and abs(expect - logit) < 2e-3 and isinstance(pred, bool)):
                mismatches += 1
            n_true += pred
            paths += use_comb
        prog_ok = all(c[4] == (int(art["prog_layer"]) if "prog_w" in art else 0) for c in calls)
        cap_ok = all(c[3] == (int(art["prog_cap"]) if "prog_w" in art else P.SAMPLING[m][1]) for c in calls)
        dec = [str(n) for n in (art["comb_feats"] if "comb_w" in art else art["cal_feats"])]
        th = {k: round(float(art[k]), 3) for k in ("cap_theta", "cap_theta_z") if k in art}
        good = mismatches == 0 and prog_ok and cap_ok and (("comb_w" in art) == bool(calls))
        ok_all &= good
        print(f"  {'OK ' if good else 'BAD'} {m.split('/')[-1][:22]:22s} {e:7s} decision={dec} combined-path "
              f"{paths}/{trials} sampled={len(calls)} True={n_true}/{trials} hook_layer_ok={prog_ok} cap_ok={cap_ok} {th}")
    return ok_all


if __name__ == "__main__":
    results = {}
    for zp in sys.argv[1:]:
        print(zp)
        results[zp] = run_zip(zp)
    print("ALL OK" if all(results.values()) else f"FAILED: {[k for k, v in results.items() if not v]}")
    sys.exit(0 if all(results.values()) else 1)
