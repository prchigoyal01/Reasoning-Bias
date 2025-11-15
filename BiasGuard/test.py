from safetensors import safe_open

with safe_open("sft_model/checkpoint-516/adapter_model.safetensors", framework="pt") as f:
    print(f.keys())