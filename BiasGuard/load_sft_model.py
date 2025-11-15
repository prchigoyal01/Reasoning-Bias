import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load LoRA adapter from HuggingFace (downloads base model + adapter automatically)
print("Loading BiasGuard LoRA adapter from HuggingFace...")
model = AutoPeftModelForCausalLM.from_pretrained(
    'ineedausername101/ANLP-BiasGuard-lora-adapter',
    device_map='auto',
    torch_dtype=torch.float16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('ineedausername101/ANLP-BiasGuard-lora-adapter')

# Test the model
test_prompt = "Should men be paid more than women for the same job?"
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Question: {test_prompt}")
print(f"Response: {response}")