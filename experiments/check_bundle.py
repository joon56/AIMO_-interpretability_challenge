import inspect, sys
import solution
from aimo_robust.predict import load_artifact
cfgs = [("Qwen/Qwen3.5-4B", "default"), ("Skywork/Skywork-OR1-Math-7B", "default"), ("allenai/Olmo-3-7B-Think", "default"),
        ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "default"), ("openai/gpt-oss-120b", "low"),
        ("openai/gpt-oss-120b", "medium"), ("openai/gpt-oss-120b", "high")]
out = []
for m, e in cfgs:
    a = load_artifact(m, e)
    if a is None:
        out.append(f"{m.split('/')[1][:6]}-{e[:3]}:none")
        continue
    mode = "full" if ("comb_w" in a and "fuse_a" in a and "prog_w" in a) else ("probe" if "comb_w" not in a else "other")
    out.append(f"{m.split('/')[1][:6]}-{e[:3]}:{mode}")
print(sys.argv[1], inspect.signature(solution.are_robust), solution.are_robust("x/unknown", ["1+1?"], reasoning_effort="default"), " ".join(out))
