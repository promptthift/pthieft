import torch
from transformers import BertTokenizer, BertModel
from torch import nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出一个浮点数
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model_name = "bert-base-uncased"  # 或 "bert-base-chinese"（适用于中文）
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name)
    mlp = SimpleMLP()
    state_dict = torch.load('shap_model.pth')
    mlp.load_state_dict(state_dict)
    inputs = tokenizer(['this is a prompt'], return_tensors="pt", 
                        padding=True, truncation=True, max_length=512)
    embeddings = bert(**inputs)
    token_embeddings = embeddings.last_hidden_state  # (batch_size, seq_len, hidden_dim)
    mean_embedding = token_embeddings.mean(dim=1)
    shap_pred = mlp(mean_embedding)
    print(shap_pred)