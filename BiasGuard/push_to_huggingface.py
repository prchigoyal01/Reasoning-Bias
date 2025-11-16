from huggingface_hub import HfApi

# Upload the entire LoRA adapter folder (48MB only - no merging needed!)
adapter_path = "rl_model"
repo_name = "RajMaheshwari/BiasGuard"

print(f"Uploading LoRA adapter to {repo_name}...")
print("This will upload ~48MB (adapter weights + tokenizer + config)")

api = HfApi()

# Create the repository if it doesn't exist
print("Creating repository...")
api.create_repo(
    repo_id=repo_name,
    repo_type="model",
    private=False,
    exist_ok=True 
)

# Upload the adapter folder
print("Uploading files...")
api.upload_large_folder(
    folder_path=adapter_path,
    repo_id=repo_name,
    repo_type="model",
    # commit_message="Upload BiasGuard LoRA adapter",
    ignore_patterns=["checkpoint-*"]
)

print(f"\n✅ LoRA adapter successfully pushed to: https://huggingface.co/{repo_name}")
print(f"\nCan use it with:")
print(f"from peft import AutoPeftModelForCausalLM")
print(f"model = AutoPeftModelForCausalLM.from_pretrained('{repo_name}', device_map='auto')")