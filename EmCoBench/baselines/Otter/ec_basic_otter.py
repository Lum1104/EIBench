import mimetypes
import os
from io import BytesIO
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm
import sys
import json
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

from otter_ai.models.otter.modeling_otter import OtterForConditionalGeneration


# Disable warnings
requests.packages.urllib3.disable_warnings()

# ------------------- Utility Functions -------------------




def get_content_type(file_path):
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type


# ------------------- Image and Video Handling Functions -------------------

def get_image(url: str) -> Union[Image.Image, list]:
    if "://" not in url:  # Local file
        content_type = get_content_type(url)
    else:  # Remote URL
        content_type = requests.head(url, stream=True, verify=False).headers.get("Content-Type")

    if "image" in content_type:
        if "://" not in url:  # Local file
            return Image.open(url)
        else:  # Remote URL
            return Image.open(requests.get(url, stream=True, verify=False).raw)
    else:
        raise ValueError("Invalid content type. Expected image or video.")


# ------------------- OTTER Prompt and Response Functions -------------------
def write_to_json(obj_dict, json_file_path):
    with open(json_file_path, 'a') as json_file:
        json_file.write(json.dumps(obj_dict) + '\n')

def get_formatted_prompt(prompt: str, in_context_prompts: list = []) -> str:
    in_context_string = ""
    for in_context_prompt, in_context_answer in in_context_prompts:
        in_context_string += f"<image>User: {in_context_prompt} GPT:<answer> {in_context_answer}<|endofchunk|>"
    return f"{in_context_string}<image>User: {prompt} GPT:<answer>"


def get_response(image_list, tokenizer, prompt: str, model=None, image_processor=None, in_context_prompts: list = []) -> str:
    input_data = image_list

    if isinstance(input_data, Image.Image):
        vision_x = image_processor.preprocess([input_data], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    elif isinstance(input_data, list):  # list of video frames
        vision_x = image_processor.preprocess(input_data, return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0)
    else:
        raise ValueError("Invalid input data. Expected PIL Image or list of video frames.")

    lang_x = model.text_tokenizer(
        [
            get_formatted_prompt(prompt, in_context_prompts),
        ],
        return_tensors="pt",
    )
    bad_words_id = tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
    generated_text = model.generate(
        vision_x=vision_x.to(model.device),
        lang_x=lang_x["input_ids"].to(model.device),
        attention_mask=lang_x["attention_mask"].to(model.device),
        max_new_tokens=512,
        num_beams=3,
        no_repeat_ngram_size=3,
        bad_words_ids=bad_words_id,
    )
    parsed_output = (
        model.text_tokenizer.decode(generated_text[0])
        .split("<answer>")[-1]
        .lstrip()
        .rstrip()
        .split("<|endofchunk|>")[0]
        .lstrip()
        .rstrip()
        .lstrip('"')
        .rstrip('"')
    )
    return parsed_output

def main(ec_data_file, output_file, model, tokenizer, image_processor, image_path):
    with open(ec_data_file, 'r') as f:
        ec_data = []
        for line in f:
            ec_data.append(json.loads(line))

    for data in tqdm(ec_data):
        for img_path, data_input in data.items():
            content = f"You are a good expert of emotion understanding. Look at the image, {data_input.lower()} Let's think step by step."
            frames_list = []
            frames = get_image(image_path+img_path)
            frames_list.append(frames)

            output = get_response(frames_list, tokenizer, content, model, image_processor)
            write_to_json({f"{img_path}": output}, output_file)


# ------------------- Main Function -------------------

if __name__ == "__main__":
    model = OtterForConditionalGeneration.from_pretrained("luodian/OTTER-Image-LLaMA7B-LA-InContext", device_map="auto")
    model.text_tokenizer.padding_side = "left"
    tokenizer = model.text_tokenizer
    image_processor = transformers.CLIPImageProcessor()
    model.eval()

    ec_data_file = "path/to/user.jsonl"
    output_file = "path/to/save.jsonl"
    image_path = "path/to/dataset/"

    main(ec_data_file, output_file, model, tokenizer, image_processor, image_path)
