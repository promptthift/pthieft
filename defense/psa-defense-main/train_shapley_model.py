import torch
from transformers import BertTokenizer, BertModel
from torch import nn, optim
from eval_token import get_freq
import json
import random
from tqdm import tqdm

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, x):
        return self.model(x)

def main():
    dict_path = 'output/stats/eff_kernelshap_dict.csv'
    with open(dict_path, 'r') as f:
        shap_dict = json.load(f)
    modifiers = list(shap_dict.keys())



    mlp = SimpleMLP(input_dim = 768)
    mlp = mlp.cuda()
    mlp.train()
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
    bs = 16
    num_epochs = 100
    criterion = nn.MSELoss()

    model_name = "bert-base-uncased" 
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name)
    bert = bert.cuda()


    for epoch in range(num_epochs):
        random.shuffle(modifiers)
        loss_total = 0
        for i in range(0, len(modifiers), bs):
            modifiers_chunk = modifiers[i:i+bs]
            with torch.no_grad():
                inputs = tokenizer(modifiers_chunk, return_tensors="pt", 
                            padding=True, truncation=True, max_length=512)
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) \
                          else v for k, v in inputs.items()}
                embeddings = bert(**inputs)
                token_embeddings = embeddings.last_hidden_state  
                mean_embedding = token_embeddings.mean(dim=1)
            shap_pred = mlp(mean_embedding)
            shap_chunk = torch.tensor([shap_dict[m] for m \
                in modifiers_chunk]).view(len(modifiers_chunk), 1).cuda()
            loss = criterion(shap_pred, shap_chunk)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_total += loss.item() * len(modifiers_chunk)



        loss_test = 0
        '''
        for i in range(0, len(modifiers_test), 10):
            modifiers_chunk = modifiers_test[i:i+bs]
            with torch.no_grad():
                inputs = tokenizer(modifiers_chunk, return_tensors="pt", 
                            padding=True, truncation=True, max_length=512)
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) \
                          else v for k, v in inputs.items()}
                embeddings = bert(**inputs)
                token_embeddings = embeddings.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                mean_embedding = token_embeddings.mean(dim=1)
                shap_pred = mlp(mean_embedding)
                shap_chunk = torch.tensor([shap_dict[m] for m \
                    in modifiers_chunk]).view(len(modifiers_chunk), 1).cuda()
                loss = criterion(shap_pred, shap_chunk)
                loss_test += loss * 10
        '''
                
        print(f'epoch {epoch} train loss {loss_total/len(modifiers)} ' + \
              f'test loss {loss_test/100}')

    torch.save(mlp.state_dict(), 'output/stats/shap_model.pth')
    return

if __name__ == '__main__':
    main()