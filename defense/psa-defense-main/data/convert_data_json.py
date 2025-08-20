import json
import argparse

def modify_image_entry(line, i):
    try:
        data = json.loads(line)
        if 'image' in data:
            data['image'] = f'{str(i).zfill(5)}.png'
        return json.dumps(data)
    except json.JSONDecodeError:
        print("Invalid JSON format:", line)
        return line

if __name__ == '__main__':
    input_file = 'data/qs.json'
    output_file = 'data/qs_adv_bb.json'
    adv_path = 'lexica_adv_bb_eps0.05'

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i >= 1000:
                break
            modified_line = modify_image_entry(line, i)
            outfile.write(modified_line + '\n')

    print(f"Finish! The updated file is: {output_file}")