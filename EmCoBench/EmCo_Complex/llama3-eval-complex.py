import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm


def write_to_json(obj_dict, json_file_path):
    with open(json_file_path, 'a') as json_file:
        json_file.write(json.dumps(obj_dict) + '\n')

def main(ec_data_file, gt_file, output_file, model_id):
    with open(ec_data_file, 'r') as f:
        ec_data = []
        for line in f:
            ec_data.append(json.loads(line))

    with open(args.gt_file) as f:
        gt = []
        for line in f:
            gt.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    for j in tqdm(range(len(ec_data))):
        for img_path1, data_input in ec_data[j].items():
            i = 0
            while True:
                content = "Your task is to assess a record aimed at comprehending an emotion and compare it against the truth label. Determine the number of potential triggers identified correctly versus those missed. Please provide your assessment in the format: {score: correct/total}, e.g. {score: 2/5} for 2 correct and 5 in total. And include an explanation of your assessment."
                for img_path2, gt_json in gt[j].items():
                    if img_path1 != img_path2:
                        print(gt_json)
                for _, label in gt_json.items():
                    content = content + f" The record is below:\n\nRecord of comprehension:\n{data_input}. Here is the ground truth label:\n\nGround Truth:\n{label}\n\nYour Assessment:"
                messages = [
                    {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer. Only say the content user wanted."},
                    {"role": "user", "content": content},
                ]

                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)

                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                outputs = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                )

                response = outputs[0][input_ids.shape[-1]:]
                output = tokenizer.decode(response, skip_special_tokens=True)
                if "{score: " in output.lower():
                    break
                if i == 4:
                    output = "{score: 0/1}"
                    break
                i += 1
            write_to_json({f"{img_path1}": output}, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Emotional comprehension Records.")
    parser.add_argument("--ec-data-file", type=str, help="Path to emotional comprehension data file (JSONL).")
    parser.add_argument("--gt-file", type=str, help="Path to ground truth data file (JSON).")
    parser.add_argument("--output-file", type=str, help="Path to output JSONL file.")
    parser.add_argument("--model-id", type=str, help="Path to the pretrained model.")
    args = parser.parse_args()

    main(args.ec_data_file, args.gt_file, args.output_file, args.model_id)
