#!/bin/bash

# AIgiSE + slime Training Script
# Usage: cd /root/slime && bash examples/aigise/run_qwen3_4B.sh

ray stop --force 2>/dev/null
sleep 2

set -ex

export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o "NV[0-9][0-9]*" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

export AIGISE_AGENT_NAME="${AIGISE_AGENT_NAME:-mock_rl_agent}"
export AIGISE_BENCHMARK_NAME="${AIGISE_BENCHMARK_NAME:-mock_debug}"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B-Instruct-2507/
   --ref-load /root/Qwen3-4B-Instruct-2507_torch_dist/
   --load /root/Qwen3-4B-Instruct-2507_slime_aigise/
   --save /root/Qwen3-4B-Instruct-2507_slime_aigise/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/aigise_data/mock_tasks.jsonl
   --input-key index
   --rollout-shuffle
   --num-rollout 20
   --rollout-batch-size 4
   --n-samples-per-prompt 2
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 8
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data mock-eval /root/aigise_data/mock_tasks.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 1024
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_aigise.generate
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
NUM_GPUS=2

# MUST be in /root/slime so train.py is found
cd /root/slime

rm -rf /root/shared/ray_temp 2>/dev/null

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    --temp-dir /root/shared/ray_temp

echo "Waiting for Ray dashboard..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8265/api/version > /dev/null 2>&1; then
        echo "Ray dashboard ready after ${i}s"
        break
    fi
    sleep 1
done
sleep 5

AIGISE_SRC="${AIGISE_SRC:-/root/aigise/src}"
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${AIGISE_SRC}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

echo "Submitting ray job from $(pwd)..."
ray job submit --address="http://127.0.0.1:8265" \
   --no-wait \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${DISTRIBUTED_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}

echo ""
echo "============================================"
echo "Job submitted. Monitor with:"
echo "  ray job list --address http://127.0.0.1:8265"
echo "  ray job logs --address http://127.0.0.1:8265 <JOB_ID> --follow"
echo "============================================"
