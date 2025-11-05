"""
This is a test that runs gpt2 locally on the GPU
Observe device log
Example output:
Model is on device: cuda:0
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
As an AI System Engineer at Fatshark, I...

(1) has developed and applied a deep learning algorithm which is capable of learning to identify objects.

(2) has been awarded the "Fatshark Prize".

(3) has been awarded the "Fats
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")

first_param = next(model.parameters())
print(f"Model is on device: {first_param.device}")

prompt = "As an AI System Engineer at Fatshark, I..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    temperature=0.8
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
