from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"  # <-- the real model now
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

prompt = "I think you are insane?"
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
