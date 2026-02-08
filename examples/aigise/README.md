# AIgiSE
This example shows slime training with AIgiSE as the agentic rollout environment. The agent runs multi-turn tool-calling trajectories for security vulnerability analysis, with LLM calls routed to slime's sglang server.

## Environment Setup

Install AIgiSE inside the container:

```bash
cd /root
git clone https://github.com/AIgiSE/AIgiSE aigise
cd aigise
pip install -e .
```

Install CodeQL (required for SeCodePLT vulnerability detection):

```bash
cd /root/aigise/src/aigise/sandbox_scripts
wget https://github.com/github/codeql-action/releases/download/codeql-bundle-v2.18.4/codeql-bundle-linux64.tar.gz
tar -xzf codeql-bundle-linux64.tar.gz codeql
rm -f codeql-bundle-linux64.tar.gz
```

Install slime (already mounted at `/root/slime` via docker-compose volume):

```bash
cd /root/slime
pip install -e .
```

## Data Preparation

```bash
cd /root/slime/examples/aigise

# Mock benchmark (no Docker/sandbox dependencies, for testing)
python aigise_mock.py \
    --local_dir /root/aigise_data \
    --dataset_path /root/aigise/src/aigise/evaluations/mock_debug/mock_test_dataset.json \
    --output_filename mock_tasks.jsonl

# SeCodePLT (vulnerability detection benchmark)
python aigise_mock.py \
    --local_dir /root/aigise_data \
    --dataset_path aigise/secodeplt \
    --dataset_split train \
    --task_subset_file /root/aigise/src/aigise/evaluations/secodeplt/metadata/successful_task_list.txt \
    --output_filename secodeplt_tasks.jsonl
```

## Model Preparation

```bash
# HF checkpoint
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --local-dir /root/Qwen3-4B-Instruct-2507

# Megatron checkpoint
cd /root/slime
source scripts/models/qwen3-4B-Instruct-2507.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B-Instruct-2507 \
    --save /root/Qwen3-4B-Instruct-2507_torch_dist
```

## Running

```bash
cd /root/slime
CUDA_VISIBLE_DEVICES=2,3 bash examples/aigise/run_qwen3_4B.sh
```

For SeCodePLT training:
```bash
AIGISE_AGENT_NAME=vul_agent_static_tools AIGISE_BENCHMARK_NAME=secodeplt \
    CUDA_VISIBLE_DEVICES=2,3 bash examples/aigise/run_qwen3_4B.sh
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `2,3` | GPUs to use |
| `AIGISE_AGENT_NAME` | `mock_rl_agent` | Agent directory under `aigise/examples/agents/` |
| `AIGISE_BENCHMARK_NAME` | `mock_debug` | Benchmark name in `aigise/evaluations/` |
| `AIGISE_SRC` | `/root/aigise/src` | Path to AIgiSE source |

Checkpoint and training hyperparameters: edit `run_qwen3_4B.sh` directly.

## Files
- `generate_with_aigise.py` — slime custom generate function (rollout entry point)
- `aigise_mock.py` — dataset to slime JSONL converter
- `run_direct.sh` — training launch script (ray start + python3 train.py)
- `run_qwen3_4B.sh` — alternative launch via ray job submit
- `test_slime_llm.py` — unit tests for SlimeLlm
