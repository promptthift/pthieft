import torch
import clip
from transformers import BertTokenizer, BertModel
from torch import nn
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = clip.load("ViT-L/14", device=device)

# CLIP similarity
def clip_similarity(sentence1, sentence2):
    text = clip.tokenize([sentence1, sentence2]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = torch.matmul(text_features[0], text_features[1]).item()
    return similarity

# load BERT and MLP
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert = BertModel.from_pretrained(model_name)

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

mlp = SimpleMLP(768)
state_dict = torch.load('shap_model.pth', map_location=torch.device('cpu'))
mlp.load_state_dict(state_dict)
mlp.eval()

# test multiple sentence without subjects
sentences = [
    "trending on artstation, digital art, octane render, ray-tracing, 4k desktop background",
    "cinematic lighting, highly detailed, trending on artstation, octane render, 4k ultra HD",
    " digital painting, trending on artstation, art nouveau style, 8k resolution"
]


all_shap_values = []
all_clip_drops = []

for sentence in sentences:
    phrases = [phrase.strip() for phrase in sentence.split(",")]
    full_clip_score = clip_similarity(sentence, sentence)

    shap_values = []
    clip_drops = []

    for phrase in phrases:
        # calculate the shap_pred
        inputs = tokenizer(phrase, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embeddings = bert(**inputs)
            mean_embedding = embeddings.last_hidden_state.mean(dim=1)
            shap_pred = abs(mlp(mean_embedding).item())  # obtain abs

        # CLIP drop calculation
        reduced_sentence = ", ".join([p for p in phrases if p != phrase])
        new_clip_score = clip_similarity(sentence, reduced_sentence)
        clip_drop = full_clip_score - new_clip_score  


        shap_values.append(shap_pred)
        clip_drops.append(clip_drop)


    all_shap_values.extend(shap_values)
    all_clip_drops.extend(clip_drops)


mean_shap = np.mean(all_shap_values)
mean_clip_drop = np.mean(all_clip_drops)


correlation = np.corrcoef(all_shap_values, all_clip_drops)[0, 1]


print(f"Mean SHAP Pred: {mean_shap:.4f}")
print(f"Mean CLIP Drop: {mean_clip_drop:.4f}")
print(f"Pearson Correlation between SHAP Pred and CLIP Drop: {correlation:.4f}")

sorted_results = sorted(zip(all_shap_values, all_clip_drops), key=lambda x: x[0], reverse=True)

print("\nSorted SHAP Pred and CLIP Drop:")
for shap, clip_drop in sorted_results:
    print(f"SHAP {shap:.4f}, CLIP Drop {clip_drop:.4f}")