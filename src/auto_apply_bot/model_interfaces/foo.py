from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load 4-bit quantized model using bitsandbytes
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype = torch.float16,
)

# Simple prompt
prompt = "Write a short, professional cover letter for a software engineering role."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs,
                         max_new_tokens=200,
                         do_sample=True,
                         temperature=0.7,
                         top_p=0.9,
                         top_k=50,
                         repetition_penalty=1.1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
