import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# 这个是文本embedding层


class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim): #分别用于指定BERT模型的类型和投影的维度

        super(BERTModel, self).__init__()
        # 创建预训练的BERT模型并加载权重
        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        # 定义投影头部，将BERT模型的输出投影到较低维度的向量空间
        self.project_head = nn.Sequential(
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )
        # freeze BERT parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # 将输入传递给BERT模型，并要求返回所有隐藏状态
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1+2+last layer,从BERT模型的输出中选择特定的隐藏状态
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        # 对选定的隐藏状态进行处理，求平均以生成嵌入表示
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # permute是为了把batch放在第一维，mean是为了把layer和seq_len维度求平均
        # 将嵌入表示传递给投影头部，进行维度投影
        embed = self.project_head(embed)
        # 返回一个字典，包含特征和投影结果
        return {'feature':output['hidden_states'],'project':embed} # batch, emb_dim


# if __name__ == '__main__':
#     print("hello")
#     bert_type = r"D:\model\biobert_base_cased_v1_2"  # 或者是您自己的模型路径
