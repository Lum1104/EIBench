import argparse
import json
from tqdm import tqdm
from openai import OpenAI
import time
import httpx
from dotenv import load_dotenv
import os
import base64
import requests

api_key = "YOUR_API_KEY"
load_dotenv()
proxy_url = os.environ.get("OPENAI_PROXY_URL")
print(proxy_url)
if not proxy_url:
    exit(0)
NUM_SECONDS_TO_SLEEP = 0.1

def write_to_json(obj_dict, json_file_path):
    with open(json_file_path, 'a') as json_file:
        json_file.write(json.dumps(obj_dict) + '\n')

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def ask_chatgpt(prompt, image_path, model="gpt-4o", temperature=0.1, max_tokens=1024):
    
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": model,
    "messages": [
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    }
    while True:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, proxies=proxies)
            #print(response)
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
    return response.json()["choices"][0]["message"]["content"]


def main(ec_data_file, gt_file, output_file):
    with open(ec_data_file, 'r') as f:
        ec_data = []
        for line in f:
            ec_data.append(json.loads(line))
    with open(output_file, 'r') as f:
        exist_data = []
        for line in f:
            exist_data += [path for path, value in json.loads(line).items()]
        


    for data in tqdm(ec_data):
        for img_path, data_input in data.items():
            content = f"You are a good expert of emotion understanding. Look at the image, {data_input}"
            output = ask_chatgpt(prompt=content, image_path="path/to/dataset/"+img_path, model="gpt-4-vision-preview")
            write_to_json({f"{img_path}": output}, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Emotional Understanding Records.")
    parser.add_argument("--ec-data-file", type=str, help="Path to emotional understanding data file (JSONL).")
    parser.add_argument("--gt-file", type=str, help="Path to ground truth data file (JSON).", default="/home/lyx/datasets/CoT-emo/dataset/full_data.json")
    parser.add_argument("--output-file", type=str, help="Path to output JSONL file.")
    args = parser.parse_args()

    main(args.ec_data_file, args.gt_file, args.output_file)
# python gpt4-score.py --ec-data-file user.jsonl --gt-file full_data.json --output-file /home/lyx/datasets/CoT-emo/evaluation/gpt4o_user_scores.jsonl
