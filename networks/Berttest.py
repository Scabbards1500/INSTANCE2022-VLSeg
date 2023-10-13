from transformers import AutoTokenizer, AutoModel

model_name = r"D:\model\biobert_base_cased_v1_2"  # 或者是您自己的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(inputs['input_ids'])
print("分割线--------------")
print(outputs['last_hidden_state'].shape)   # torch.Size([1, 8, 768])
print(outputs['pooler_output'].shape)       # torch.Size([1, 768])
print(outputs)
