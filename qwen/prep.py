import os

from transformers import AutoModelForCausalLM, AutoTokenizer

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_name = "QWen/QWen2.5-0.5B"

model = AutoModelForCausalLM.from_pretrained(model_name)

print(model.config)
print(model.generation_config)

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          padding_side="left",
                                          add_eos_token=False)
print(tokenizer.padding_side, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
print(hasattr(tokenizer, "pad_token_id"))

text = ["this is a test.", "this is another test. is it ok?"]
max_len = 16
model_inputs = tokenizer(text, max_length=max_len-1,
                                 padding="max_length", truncation=True, add_special_tokens=True)
print(model_inputs)


def check_add_eos(tokenized_inputs, tokenizer):
    if tokenized_inputs.input_ids[0][-1] != tokenizer.eos_token_id:
        for idx in range(len(tokenized_inputs.input_ids)):
            tokenized_inputs.input_ids[idx].append(tokenizer.eos_token_id)
            tokenized_inputs.attention_mask[idx].append(1)


check_add_eos(model_inputs, tokenizer)
print(model_inputs)

