import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer

# Load base model and LoRA weights
model = AutoModelForCausalLM.from_pretrained("/home/liuxf/hdisk/llm/ALMA-7B-Pretrain", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, "/home/liuxf/hdisk/llm/ALMA-7B-Pretrain-LoRA")
tokenizer = LlamaTokenizer.from_pretrained("/home/liuxf/hdisk/llm/ALMA-7B-Pretrain", padding_side='left')

# Add the source sentence into the prompt template
prompt="Translate this from Chinese to English:\nChinese: 我爱机器翻译。\nEnglish:"
input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.cuda()

# Translation
with torch.no_grad():
    generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=20, do_sample=True, temperature=0.6, top_p=0.9)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(outputs)