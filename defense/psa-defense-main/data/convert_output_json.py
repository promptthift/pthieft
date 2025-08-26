import json
import pandas as pd

if __name__ == '__main__':
    jsonl_file = 'output/PS_results/answer-file-our_bb.jsonl'
    csv_file = 'output/PS_results/llava_results_bb.csv'


    data = []
    with open(jsonl_file, 'r') as f:
        for line in f:

            record = json.loads(line)
            

            a_value = record.get('text', None)

            data.append({'inferred_prompt': a_value, })


    df = pd.DataFrame(data)


    df.to_csv(csv_file, index=False)

    print(f"CSV file saved to {csv_file}")