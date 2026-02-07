# AIgiSE
This example shows slime training with AIgiSE as the agentic rollout environment. The agent runs multi-turn tool-calling trajectories for security vulnerability analysis, with LLM calls routed to slime sglang server.

## Environment Setup
Use the `zhuzilin/slime:latest` image:

```bash
cd /root/
git clone <aigise-repo-url> aigise
cd aigise
pip install -e . --no-deps
```

## Data Preparation
Generate mock task data:

```bash
cd /root/slime/examples/aigise
python aigise_mock.py --output_dir /root/aigise_data --output_filename mock_tasks.jsonl
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
bash examples/aigise/run_direct.sh
```

Select GPUs via environment variable (defaults to 2,3):
```bash
CUDA_VISIBLE_DEVICES=4,5 bash examples/aigise/run_direct.sh
```

## Configuration
- Agent: set `AIGISE_AGENT_NAME` env var (default: `mock_rl_agent`)
- Benchmark: set `AIGISE_BENCHMARK_NAME` env var (default: `mock_debug`)
- Checkpoint paths: edit `run_direct.sh`

## Files
- `generate_with_aigise.py` — slime custom generate function (rollout entry point)
- `aigise_mock.py` — mock data generator
- `run_direct.sh` — training launch script
- `run_qwen3_4B.sh` — alternative launch via ray job submit
- `test_slime_llm.py` — unit tests for SlimeLlm
