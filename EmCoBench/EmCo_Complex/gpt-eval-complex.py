import argparse
import json
from tqdm import tqdm
from openai import OpenAI
import time


NUM_SECONDS_TO_SLEEP = 0.1

def write_to_json(obj_dict, json_file_path):
    with open(json_file_path, 'a') as json_file:
        json_file.write(json.dumps(obj_dict) + '\n')

def ask_chatgpt(prompt, model="gpt-4", temperature=0.1, max_tokens=512):
    client = OpenAI(api_key="YOUR_API_KEY")
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer. Only say the content user wanted.'
                }, {
                    'role': 'user',
                    'content': prompt,
                }],
                temperature=temperature,  # TODO: figure out which temperature is best
                max_tokens=max_tokens,
            )
            if "{score" not in response.choices[0].message.content:
                raise
            break
        except Exception as e:
            print(e)
        time.sleep(0.1)
    return response.choices[0].message.content


def main(ec_data_file, gt_file, output_file):
    with open(ec_data_file, 'r') as f:
        ec_data = []
        for line in f:
            ec_data.append(json.loads(line))

    with open(args.gt_file) as f:
        gt = []
        for line in f:
            gt.append(json.loads(line))

    for i in tqdm(range(len(ec_data))):
        for img_path1, data_input in ec_data[i].items():
            content = "Your task is to assess a record aimed at comprehension an emotion and compare it against the truth label. Determine the number of potential triggers identified correctly versus those missed. Please provide your assessment in the format: {score: correct/total}, e.g. {score: 2/5} for 2 correct and 5 in total from Ground Truth. And include an explanation of your assessment."
            for _, gt_json in gt[i].items():
                for _, label in gt_json.items():
                    content = content + f" The record is below:\n\nRecord of comprehension:\n{data_input}. Here is the ground truth label:\n\nGround Truth:\n{label}\n\nYour Assessment:"
                    output = ask_chatgpt(prompt=content, model="gpt-3.5-turbo")
                    write_to_json({f"{img_path1}": output}, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Emotional comprehension Records.")
    parser.add_argument("--ec-data-file", type=str, help="Path to emotional comprehension data file (JSONL).")
    parser.add_argument("--gt-file", type=str, help="Path to ground truth data file (JSON).")
    parser.add_argument("--output-file", type=str, help="Path to output JSONL file.")
    args = parser.parse_args()

    main(args.ec_data_file, args.gt_file, args.output_file)
# python gpt-eval-complex.py --ec-data-file experiments/llava7b_complex.jsonl --gt-file ec_complex.jsonl
