"""Sample n solutions per (config, problem) with vLLM using the label-generation sampling settings.

Dev-only data generation for solve-rate targets. Resumable: problems already in the output are skipped.
Usage: python gen_solve.py --model openai/gpt-oss-120b --efforts low,medium,high --outdir runs/gen
Writes <outdir>/<safe_cfg>.jsonl (+ .texts.jsonl.gz with raw generations).
Prompts are tokenized with our round-trip-checked tokenizer and passed as token ids (vLLM's AutoTokenizer
strips spaces for DeepSeek-R1-0528-Qwen3-8B in transformers 5.x); outputs are decoded with the same tokenizer.
"""

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aimo_robust.modeling import _roundtrip_ok, load_tokenizer  # noqa: E402
from aimo_robust.prompting import build_prompt, is_gpt_oss, parse_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--efforts", default="default")
    ap.add_argument("--problems", default=str(Path(__file__).resolve().parents[1] / "data/pool/problems.jsonl"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="shard: problem slice start")
    ap.add_argument("--end", type=int, default=0, help="shard: problem slice end (0 = all)")
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--attn", default="", help="vLLM attention backend override, e.g. TRITON_ATTN")
    args = ap.parse_args()

    from transformers import GenerationConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = load_tokenizer(args.model, local_files_only=False)
    # vLLM tokenizes with AutoTokenizer: skip its init only where that tokenizer is broken (DeepSeek); multimodal
    # checkpoints (Qwen3.5) need it for their processor.
    from transformers import AutoTokenizer
    skip_tok = not _roundtrip_ok(AutoTokenizer.from_pretrained(args.model))
    eos = {tok.eos_token_id}
    try:
        g = GenerationConfig.from_pretrained(args.model).eos_token_id
        eos.update([g] if isinstance(g, int) else (g or []))
    except OSError:
        pass
    if is_gpt_oss(args.model):
        eos.update(tok.get_vocab()[t] for t in ("<|return|>", "<|call|>"))
    eos = sorted(e for e in eos if e is not None)
    print(f"tokenizer {type(tok).__name__}, stop ids {eos}, skip vLLM tokenizer init: {skip_tok}", flush=True)

    problems = [json.loads(line) for line in open(args.problems, encoding="utf-8")]
    if args.limit:
        problems = problems[: args.limit]
    sharded = bool(args.start or args.end)
    problems = problems[args.start : args.end or None]
    llm = None
    CHUNK = args.chunk  # chunked so partial progress survives interruptions

    for effort in args.efforts.split(","):
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{args.model}__{effort}")
        out = Path(args.outdir) / (stem + (f".part{args.start}" if sharded else "") + ".jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(line)["pid"] for line in out.open(encoding="utf-8")} if out.exists() else set()
        todo = [p for p in problems if p["pid"] not in done]
        print(f"{args.model} {effort}: {len(todo)} todo / {len(problems)}", flush=True)
        if not todo:
            continue
        if llm is None:
            llm = LLM(model=args.model, max_model_len=args.max_tokens + 4096, gpu_memory_utilization=args.gpu_mem,
                      enable_prefix_caching=True, tensor_parallel_size=args.tp, skip_tokenizer_init=skip_tok,
                      **({"attention_backend": args.attn} if args.attn else {}))
        prompts = [TokensPrompt(prompt_token_ids=tok(build_prompt(tok, args.model, p["problem"], effort),
                                                     add_special_tokens=False)["input_ids"]) for p in todo]
        sp = SamplingParams(n=args.n, temperature=1.0, top_k=40, top_p=0.95, max_tokens=args.max_tokens,
                            detokenize=False, stop_token_ids=eos, seed=0)
        t0 = time.time()
        texts_path = out.with_suffix(".texts.jsonl.gz")
        for i in range(0, len(todo), CHUNK):
            batch, bprompts = todo[i : i + CHUNK], prompts[i : i + CHUNK]
            results = llm.generate(bprompts, sp)
            ntok = 0
            with out.open("a", encoding="utf-8") as fh, gzip.open(texts_path, "at", encoding="utf-8") as tf:
                for p, r in zip(batch, results):
                    samples = []
                    for o in r.outputs:
                        ntok += len(o.token_ids)
                        text = tok.decode(o.token_ids, skip_special_tokens=False)
                        samples.append({"ans": parse_answer(text), "len": len(o.token_ids), "finish": o.finish_reason})
                        tf.write(json.dumps({"pid": p["pid"], "text": text}, ensure_ascii=False) + "\n")
                    fh.write(json.dumps({"pid": p["pid"], "model": args.model, "effort": effort,
                                         "prompt_len": len(r.prompt_token_ids), "samples": samples}) + "\n")
            print(f"{effort} chunk {i // CHUNK}: {len(batch)} problems, {ntok} tokens, "
                  f"{ntok / (time.time() - t0):.0f} tok/s cumulative-chunk, elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
