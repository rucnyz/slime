#!/bin/bash
# Direct run: runs train.py directly (alternative to ray job submit via run_qwen3_4B.sh)
# Usage: bash /root/slime/examples/aigise/run_direct.sh

set -ex

ulimit -n 655360
export PYTHONBUFFERED=16
export CUDA_DEVICE_MAX_CONNECTIONS=1
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

export AIGISE_AGENT_NAME="${AIGISE_AGENT_NAME:-mock_rl_agent}"
export AIGISE_BENCHMARK_NAME="${AIGISE_BENCHMARK_NAME:-mock_debug}"

# PYTHONPATH for generate_with_aigise and aigise imports
AIGISE_SRC="${AIGISE_SRC:-/root/aigise/src}"
export PYTHONPATH="/root/Megatron-LM/:${SCRIPT_DIR}:${AIGISE_SRC}:${PYTHONPATH}"

cd /root/slime

# Clean up and start fresh ray cluster
ray stop --force 2>/dev/null
sleep 2
rm -rf /root/shared/ray_temp 2>/dev/null

NUM_GPUS=2
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 \
    --temp-dir /root/shared/ray_temp

echo "Waiting for Ray..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8265/api/version > /dev/null 2>&1; then
        echo "Ray ready after ${i}s"
        break
    fi
    sleep 1
done
sleep 3

# Run train.py directly (NOT via ray job submit)
echo "Starting train.py directly..."
python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --rollout-num-gpus ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-4B-Instruct-2507/ \
   --ref-load /root/Qwen3-4B-Instruct-2507_torch_dist/ \
   --load /root/Qwen3-4B-Instruct-2507_slime_aigise/ \
   --save /root/Qwen3-4B-Instruct-2507_slime_aigise/ \
   --save-interval 20 \
   --prompt-data /root/aigise_data/mock_tasks.jsonl \
   --input-key index \
   --rollout-shuffle \
   --num-rollout 20 \
   --rollout-batch-size 4 \
   --n-samples-per-prompt 2 \
   --rollout-max-response-len 1024 \
   --rollout-temperature 1 \
   --global-batch-size 8 \
   --balance-data \
   --optimizer adam \
   --lr 1e-6 \
   --lr-decay-style constant \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.98 \
   --advantage-estimator grpo \
   --use-kl-loss \
   --kl-loss-coef 0.00 \
   --kl-loss-type low_var_kl \
   --entropy-coef 0.00 \
   --eps-clip 0.2 \
   --eps-clip-high 0.28 \
   --tensor-model-parallel-size 2 \
   --sequence-parallel \
   --pipeline-model-parallel-size 1 \
   --context-parallel-size 1 \
   --expert-model-parallel-size 1 \
   --expert-tensor-parallel-size 1 \
   --recompute-granularity full \
   --recompute-method uniform \
   --recompute-num-layers 1 \
   --use-dynamic-batch-size \
   --max-tokens-per-gpu 9216 \
   --eval-interval 10 \
   --eval-prompt-data mock-eval /root/aigise_data/mock_tasks.jsonl \
   --n-samples-per-eval-prompt 1 \
   --eval-max-response-len 1024 \
   --eval-top-k 1 \
   --rollout-num-gpus-per-engine 1 \
   --sglang-mem-fraction-static 0.7 \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --accumulate-allreduce-grads-in-fp32 \
   --attention-softmax-in-fp32 \
   --attention-backend flash \
   --custom-generate-function-path generate_with_aigise.generate
