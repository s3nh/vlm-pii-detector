#!/bin/bash
# start_vllm.sh - Production deployment script

# Configuration
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2-VL-7B-Instruct"}  # Change to local path if offline
PORT=${PORT:-8000}
QUANTIZATION=${QUANTIZATION:-""}  # Set to "awq" or "gptq" for quantized models
TP_SIZE=${TP_SIZE:-1}  # Tensor parallelism (set to number of GPUs)

# Optional: Model download for offline environments
# If you have internet on another machine:
# huggingface-cli download $MODEL_PATH --local-dir /path/to/local/model
# Then set MODEL_PATH=/path/to/local/model

# Build command
CMD="python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 16 \
    --max-model-len 4096"

if [ -n "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

# For CPU-only environments (very slow):
# CMD="$CMD --device cpu"

echo "Starting VLLM server..."
echo "Command: $CMD"
exec $CMD
