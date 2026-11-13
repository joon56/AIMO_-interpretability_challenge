#!/usr/bin/env bash
# One-time pod setup. Two venvs:
#   /workspace/venv-rt   : mirrors the competition runtime (torch 2.12.1, transformers 5.13.0, ...)
#   /workspace/venv-vllm : fast sampling for solve-rate labels (dev only, never shipped)
set -euo pipefail
cd /workspace
export HF_HOME=/workspace/hf
mkdir -p "$HF_HOME" logs

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x /workspace/venv-tools/bin/hf ] || { uv venv /workspace/venv-tools --python 3.12 && uv pip install --python /workspace/venv-tools/bin/python "huggingface-hub[hf_xet]"; }

if [ ! -x /workspace/venv-rt/bin/python ]; then
  uv venv /workspace/venv-rt --python 3.12
  ok=""
  for idx in cu132 cu130 cu129 cu128; do
    if uv pip install --python /workspace/venv-rt/bin/python "torch==2.12.1" \
         --index-url "https://download.pytorch.org/whl/$idx"; then ok=$idx; break; fi
  done
  echo "venv-rt torch index: $ok"
  uv pip install --python /workspace/venv-rt/bin/python \
    "accelerate==1.13.0" "huggingface-hub==1.22.0" "joblib==1.5.3" "pandas==3.0.3" \
    "safetensors==0.8.0" "scikit-learn==1.8.0" "tokenizers==0.22.2" "transformers==5.13.0"
fi

if [ ! -x /workspace/venv-vllm/bin/python ]; then
  uv venv /workspace/venv-vllm --python 3.12
  uv pip install --python /workspace/venv-vllm/bin/python vllm pandas
fi

# dev-only extras: gpt-oss native MXFP4 needs kernels in [0.15.2, 0.16) for transformers 5.13; parquet; vLLM JIT
uv pip install --python /workspace/venv-rt/bin/python "kernels>=0.15.2,<0.16" pyarrow
uv pip install --python /workspace/venv-vllm/bin/python ninja
/workspace/venv-rt/bin/python -c "import torch, transformers; print('rt', torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
/workspace/venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, torch.__version__)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
