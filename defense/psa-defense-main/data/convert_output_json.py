import json
import pandas as pd

if __name__ == '__main__':
    jsonl_file = 'output/PS_results/answer-file-our_bb.jsonl'
    csv_file = 'output/PS_results/llava_results_bb.csv'

    # 读取jsonl文件并提取数据
    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            # 解析每行json数据
            record = json.loads(line)
            
            # 提取所需的字段
            a_value = record.get('text', None)
            # b_value = record.get('b', None)
            # c_value = record.get('c', None)
            
            # 将提取的值添加到数据列表
            # data.append({'a': a_value, 'b': b_value, 'c': c_value})
            data.append({'inferred_prompt': a_value, })

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)

    # 保存为 CSV 文件
    df.to_csv(csv_file, index=False)

    print(f"CSV file saved to {csv_file}")