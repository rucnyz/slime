# Tau bench 
This example shows slime training in an agentic multi-turn tool use environment. 


## Environment Setup 
Use the `zhuzilin/slime:latest` image and initialize the environment required for Search-R1:
    
```bash
cd /root/
git clone https://github.com/THUDM/slime.git
cd slime
pip install -e .
# for aigise

```

Initialize the Qwen3-32B model needed for tool use:

```bash
export BASE_DIR=/root/.cache
# hf checkpoint
huggingface-cli download Qwen/Qwen3-32B --local-dir $BASE_DIR/Qwen3-32B

# mcore checkpoint
cd /root/slime
source scripts/models/qwen3-32B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint $BASE_DIR/Qwen3-32B \
    --save $BASE_DIR/Qwen3-32B_torch_dist
```

## Running the Script

And run:

```bash
cd /root/slime
export BASE_DIR=/root/.cache
export MASTER_ADDR=127.0.0.1
export WANDB_KEY=your_wandb_key
bash examples/aigise/run_qwen3_32B.sh
```

Remember to run `ulimit -n 65536` before starting the training.