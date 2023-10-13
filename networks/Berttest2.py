import torch
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained(r"D:\model\biobert_base_cased_v1_2")
model = BertModel.from_pretrained(r"D:\model\biobert_base_cased_v1_2")

sentence = "今天天气怎么样？"
# add_special_tokens=True 则前后会分别加上<SOS> <EOS>的embedding
input_ids = tokenizer.encode(sentence, add_special_tokens=True)
input_ids = torch.tensor([input_ids])
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
# last_hidden_states.shape is (1, 8, 768)
