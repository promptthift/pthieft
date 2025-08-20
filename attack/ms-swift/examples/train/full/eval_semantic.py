import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

def clip_similarity(sentence1, sentence2):
    # 将文本转换为张量
    text = clip.tokenize([sentence1, sentence2]).to(device)
    
    # 计算文本嵌入
    with torch.no_grad():
        text_features = model.encode_text(text)

    # 归一化嵌入向量
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 计算余弦相似度
    similarity = torch.matmul(text_features[0], text_features[1]).item()
    
    return similarity

csv = False
if csv:
    file_path = "/fred/oz337/zdeng/prompt-stealing-attack-image/output/PS_results/prompt_stealer_results.csv"  # 请替换为你的实际文件路径
    df = pd.read_csv(file_path)
    df['semantic_score'] = df.apply(lambda row: clip_similarity(row['prompt'], row['inferred_prompt']), axis=1)
    
else: 
    file_path = "/fred/oz339/zdeng/output/sft_1/v2-20250305-140119/checkpoint-17000/infer_result/20250306-102328.jsonl"  
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    df = pd.DataFrame(data)
    df['semantic_score'] = df.apply(lambda row: clip_similarity(row['response'], row['labels']), axis=1)



average_jaccard = df['jaccard_score'][:43].mean()
average_semantic = df['semantic_score'][:43].mean()
# average_modifier = df['semantic_sim'].mean()
print(f"Average Jaccard Score: {average_jaccard:.4f}")
print(f"Average semantic_sim: {average_semantic:.4f}")