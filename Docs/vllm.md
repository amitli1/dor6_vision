# Download:
## gemma-3-4b-it:
``` huggingface-cli download google/gemma-3-4b-it --local-dir /home/amitli/repo/VisionModels/models/models/gemma3-4b-it/ ```

## gemma-3-27b-it:
``` huggingface-cli download google/gemma-3-27b-it --local-dir /home/amitli/repo/VisionModels/models/models/gemma-3-27b-it/ ```


# VLM:
```
sudo docker run --runtime=nvidia     -e NVIDIA_VISIBLE_DEVICES=all     -v /home/amitli/repo/VisionModels/models/models/gemma3-4b-it/:/model_path     -p 9000:9000     --ipc=host     vllm/vllm-openai:latest     --model /model_path     --port 9000     --dtype bfloat16     --trust-remote-code     --max-model-len 8192
```

# Port forwarding:
```
ssh -L 9000:localhost:9000 amitli@T-P-SHIRYY2-L-RF
```

# Test (after port forwarding):
```
curl http://T-P-SHIRYY2-L-RF:9000/v1/models
```