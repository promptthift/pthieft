import string
import pandas as pd
import json
import re
from collections import Counter
def calculate_jaccard_similarity(str1, str2):
    """计算两个字符串的Jaccard相似度"""
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def jaccard_score(text1: str, text2: str) -> float:
    """
    计算去除标点符号后的 Jaccard 相似度。
    :param text1: 第一个文本字符串
    :param text2: 第二个文本字符串
    :return: Jaccard 相似度得分 (0 到 1 之间)
    """
    translator = str.maketrans('', '', string.punctuation)
    tokens1 = set(text1.translate(translator).lower().split())
    tokens2 = set(text2.translate(translator).lower().split())
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union != 0 else 0.0

def jaccard_score_multiset(sentence1, sentence2):
    """
    Calculate the Jaccard score between two sentences based on their words.
    
    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
        
    Returns:
        float: The Jaccard score between the two sentences.
    """
    sentence2 = sentence2.split(',')[0] 
    sentence1 = sentence1.split(',')[0] 
    sentence1 = re.sub(r'[^\w\s]', '', sentence1.lower())
    sentence2 = re.sub(r'[^\w\s]', '', sentence2.lower())
    # Tokenize sentences into words
    
    words1 = Counter(sentence1.split())
    words2 = Counter(sentence2.split())
    
    # Calculate the intersection and union
    intersection = sum((words1 & words2).values())
    union = sum((words1 | words2).values()) 
    
    # del sentence1, sentence2, words1, words2, intersection, union
    # gc.collect()  
    return intersection / union if union > 0 else 0.0

def calculate_token_accuracy(text1: str, text2: str) -> float:
    """
    计算两个文本之间的token准确率
    
    Args:
        text1 (str): 第一个文本字符串（预测文本）
        text2 (str): 第二个文本字符串（参考文本）
        
    Returns:
        float: token准确率 (0 到 1 之间)
    """
    # 预处理文本：转换为小写并移除标点符号
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    # 将文本分割成token
    intersection = len(set1.intersection(set2))
    

    total_tokens = len(set2)
    
    return intersection / total_tokens if total_tokens > 0 else 0.0

# 这个是词组级别的
# def jaccard_score(text1: str, text2: str) -> float:
#     """
#     计算去除标点符号后的 Jaccard 相似度（按逗号分割词组）。
#     :param text1: 第一个文本字符串
#     :param text2: 第二个文本字符串
#     :return: Jaccard 相似度得分 (0 到 1 之间)
#     """
#     # 去除标点符号并转换为小写
#     # translator = str.maketrans('', '', string.punctuation)
#     # text1_cleaned = text1.translate(translator).lower()
#     # text2_cleaned = text2.translate(translator).lower()
    
#     # 按逗号分割成词组
#     tokens1 = set([phrase.strip() for phrase in text1.split(',') if phrase.strip()])
#     tokens2 = set([phrase.strip() for phrase in text2.split(',') if phrase.strip()])
#     # print(tokens1)
#     # print(tokens2)
#     # 计算 Jaccard 评分
#     intersection = len(tokens1 & tokens2)
#     union = len(tokens1 | tokens2)
#     return intersection / union if union != 0 else 0.0


# text1 = "portrait of a boxer, oil on wood, by adam sprys, greg rutkow safwat abyad, artstatus, art Après, highly detailed, cinematic lighting, 8 k octane "
# text2 = "realistic oil painting of will smith punching a rock, detailed, by rembrandt van rijn, lisa frank, hr giger, beksinski, anato finnstark!!, 8 k resolution, beautiful lighting, studio light, extremely detailed, establishing shot, realistic materials, hyperrealistic "
# print(jaccard_score(text1,text2))
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# 加载预训练模型（可以选择更强的模型，如 "all-MiniLM-L6-v2"）
# model = SentenceTransformer('all-MiniLM-L6-v2',cache_dir = '/fred/oz337/zdeng/prompt_stealing_ours/.cache')

# def semantic_distance(sentence1, sentence2):
#     # 计算句子的嵌入向量
#     embedding1 = model.encode(sentence1, convert_to_tensor=True)
#     embedding2 = model.encode(sentence2, convert_to_tensor=True)
    
#     # 计算余弦相似度
#     similarity = cosine_similarity([embedding1.cpu().numpy()], [embedding2.cpu().numpy()])[0][0]
    
#     # 语义距离 = 1 - 相似度
#     distance = 1 - similarity
#     return distance

csv = False
if csv:
    file_path = "/fred/oz337/zdeng/prompt-stealing-attack-image/output/PS_results/prompt_stealer_results.csv"  # 请替换为你的实际文件路径
    df = pd.read_csv(file_path)
    df['jaccard_score'] = df.apply(lambda row: jaccard_score_multiset(row['prompt'], row['inferred_prompt']), axis=1)
    df['token_accuracy'] = df.apply(lambda row: calculate_token_accuracy(row['prompt'], row['inferred_prompt']), axis=1)
else: 
    file_path = "/fred/oz339/zdeng/output/lora/BLUE_JAC_TOKEN/v3-20250305-161704/checkpoint-106000-merged/infer_result/20250417-160508.jsonl"  
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    df = pd.DataFrame(data)
    df['jaccard_score'] = df.apply(lambda row: calculate_jaccard_similarity(row['response'], row['labels']), axis=1)
    df['token_accuracy'] = df.apply(lambda row: calculate_token_accuracy(row['response'], row['labels']), axis=1)

average_jaccard = df['jaccard_score'][:43].mean()
average_token_accuracy = df['token_accuracy'][:43].mean()
print(f"Average Jaccard Score: {average_jaccard:.4f}")
print(f"Average Token Accuracy: {average_token_accuracy:.4f}")