"""Print architecture/precision/context/revision of every cached checkpoint (for the report and sanity checks)."""

import glob
import json
import os

for d in sorted(glob.glob("/workspace/hf/hub/models--*")):
    name = d.split("models--")[1].replace("--", "/")
    rev = os.listdir(d + "/snapshots")[0]
    snap = f"{d}/snapshots/{rev}"
    cfg = json.load(open(f"{snap}/config.json"))
    tc = cfg.get("text_config") or cfg
    gen = json.load(open(f"{snap}/generation_config.json")) if os.path.exists(f"{snap}/generation_config.json") else {}
    size = sum(os.path.getsize(os.path.realpath(f)) for f in glob.glob(f"{snap}/*.safetensors"))
    lt = tc.get("layer_types") or []
    print(json.dumps({
        "model": name, "rev": rev[:12], "arch": (cfg.get("architectures") or [None])[0],
        "dtype": cfg.get("torch_dtype") or cfg.get("dtype") or tc.get("torch_dtype") or tc.get("dtype"),
        "quant": (cfg.get("quantization_config") or {}).get("quant_method"),
        "layers": tc.get("num_hidden_layers"), "hidden": tc.get("hidden_size"),
        "heads": tc.get("num_attention_heads"), "kv_heads": tc.get("num_key_value_heads"),
        "experts": tc.get("num_local_experts") or tc.get("num_experts"),
        "experts_per_tok": tc.get("num_experts_per_tok"),
        "max_pos": tc.get("max_position_embeddings"), "sliding_window": tc.get("sliding_window"),
        "layer_types": {t: lt.count(t) for t in sorted(set(lt))} if lt else None,
        "weights_GB": round(size / 1e9, 1),
        "gen_defaults": {k: gen.get(k) for k in ("temperature", "top_p", "top_k", "max_new_tokens") if k in gen},
    }))
