# Download:
## gemma-3-4b-it:
``` huggingface-cli download google/gemma-3-4b-it --local-dir /home/amitli/repo/VisionModels/models/models/gemma3-4b-it/ ```

## gemma-3-27b-it:
``` huggingface-cli download google/gemma-3-27b-it --local-dir /home/amitli/repo/VisionModels/models/models/gemma-3-27b-it/ ```


# VLM:
```
sudo docker run --runtime=nvidia     -e NVIDIA_VISIBLE_DEVICES=all     -v /home/amitli/repo/VisionModels/models/models/gemma3-4b-it/:/model_path     -p 9000:9000     --ipc=host     vllm/vllm-openai:latest     --model /model_path     --port 9000     --dtype bfloat16     --trust-remote-code     --max-model-len 8192
```

```
sudo docker run --runtime=nvidia     -e NVIDIA_VISIBLE_DEVICES=all      -v ~/.cache/huggingface:/root/.cache/huggingface     -p 9000:9000     --ipc=host     vllm/vllm-openai:latest     --model google/gemma-4-E2B-it     --port 9000     --dtype bfloat16     --trust-remote-code     --max-model-len 8192 --gpu-memory-utilization 0.90
```

# VLLM (molmo2):
```
sudo docker run --runtime=nvidia     -e NVIDIA_VISIBLE_DEVICES=all     -v ~/.cache/huggingface:/root/.cache/huggingface     -p 9100:9100     --ipc=host     vllm/vllm-openai:latest     --model allenai/Molmo2-4B     --trust-remote-code     --max-num-batched-tokens 4096 --max-model-len 4096 --gpu-memory-utilization 0.95 --dtype bfloat16
```

# concepts:
1. --max-model-len 8192 -> maximum context length per request: (prompt + output) <= 8192
2. --max-num-batched-tokens 32768 -> aximum total number of tokens that vLLM is allowed to process together in one scheduling step across all requests currently being served
3. max-num-batched-tokens >= max-model-len

# Port forwarding:
```
ssh -L 9000:localhost:9000 amitli@T-P-SHIRYY2-L-RF
```

# Test (after port forwarding):
```
curl http://T-P-SHIRYY2-L-RF:9000/v1/models
```