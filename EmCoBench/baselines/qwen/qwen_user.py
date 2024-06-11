import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
from tqdm import tqdm
torch.manual_seed(1234)

def write_to_json(obj_dict, json_file_path):
    with open(json_file_path, 'a') as json_file:
        json_file.write(json.dumps(obj_dict) + '\n')

def main(args):
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True).eval()

    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)

    with open(args.input_json) as f:
        data_jsons = []
        for line in f:
            data_jsons.append(json.loads(line))

    for data_json in tqdm(data_jsons):
        for img_path, question in data_json.items():
            query = tokenizer.from_list_format([
                {'image': args.image_path + img_path}, # Either a local path or an url
                {'text': f'You are a good expert of emotion understanding. Look at the image, {question}'},
            ])
            response, _ = model.chat(tokenizer, query=query, history=None)
            write_to_json({img_path: response}, args.output_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and generate responses.")
    parser.add_argument('--model-path', type=str, required=True, help="Path to the model")
    parser.add_argument('--image-path', type=str, required=True, help="Path to the model")
    parser.add_argument('--input-json', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--output-json', type=str, required=True, help="Path to the output JSON file")
    args = parser.parse_args()
    main(args)
# python qwen_user.py --model_path path/to/model --input_json path/to/user.jsonl --output_json path/to/output.jsonl --image-path path/to/dataset
