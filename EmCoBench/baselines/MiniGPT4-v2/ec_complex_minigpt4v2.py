import argparse
import glob
import os
import random
from collections import defaultdict

import cv2
import re

import numpy as np
from PIL import Image
import torch
import html
import gradio as gr
import json

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from tqdm import tqdm

class Chat:
    def __init__(self, model, vis_processor, device='cuda:0', stopping_criteria=None):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor

        self.stopping_criteria = stopping_criteria


    def answer_prepare(self, prompt, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
                        repetition_penalty=1.05, length_penalty=1, temperature=1.0, max_length=2000):
        # prompt = f"<s>[INST] <Img><ImageHere></Img> [grounding] please describe this image? [/INST]"
        # print('prompt:', prompt)
        # print('img_list:', img_list)
        embs = self.model.get_context_emb(prompt, img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                    'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=None,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=float(temperature),
        )
        return generation_kwargs
    
    def model_generate(self, *args, **kwargs):
        # print("Positional arguments (args):", args)
        # print("Keyword arguments (kwargs):", kwargs)
        # for 8 bit and 16 bit compatibility
        with self.model.maybe_autocast():
            output = self.model.llama_model.generate(*args, **kwargs)
        return output
    
    def stream_answer(self, prompt, img_list, **kargs):
        # print('stream_answer img shape: ', img_list[0].shape)
        # random_tensor = torch.randn(1, 1, 4096).to('cuda:0')
        # img_list[0] = torch.cat((img_list[0], random_tensor), dim=1)
        # print('merged_tensor shape: ', img_list[0].shape)

        generation_kwargs = self.answer_prepare(prompt, img_list, **kargs)
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        streamer = TextIteratorStreamer(self.model.llama_tokenizer, skip_special_tokens=True)
        generation_kwargs['streamer'] = streamer
        from threading import Thread
        thread = Thread(target=self.model_generate, kwargs=generation_kwargs)
        thread.start()
        return streamer
    
    def escape_markdown(self, text):
        # List of Markdown special characters that need to be escaped
        md_chars = ['<', '>']

        # Escape each special character
        for char in md_chars:
            text = text.replace(char, '\\' + char)

        return text

    def encode_img(self, img_list):
        img_emb_list = []
        for image in img_list:
            if isinstance(image, str):  # is a image path
                raw_image = Image.open(image).convert('RGB')
                image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            elif isinstance(image, Image.Image):
                raw_image = image
                image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
            elif isinstance(image, torch.Tensor):
                if len(image.shape) == 3:
                    image = image.unsqueeze(0)
                image = image.to(self.device)

            image_emb, _ = self.model.encode_img(image)
            img_emb_list.append(image_emb)
        return img_emb_list


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    parser.add_argument("--img-path", default='/home/czb/datasets/C2F-SelfAsk.json', required=True)
    parser.add_argument("--out-json", default='/home/czb/datasets/ablation_minigpt4_C2F.json', required=True)
    args = parser.parse_args()
    return args


def Generate4img(chat, image_path, prompt):
    image = Image.open(image_path)
    image = image.convert("RGB")
    img_list=[image]
    if len(img_list) > 0:
        if not isinstance(img_list[0], torch.Tensor):
            img_list = chat.encode_img(img_list)

    prefix = f"<s>[INST] <Img><ImageHere></Img> "
    suffix = f" [/INST]"
    prompt = prefix + prompt + suffix

    streamer = chat.stream_answer(prompt=prompt,
                            img_list=img_list,
                            temperature=0.6,
                            max_new_tokens=500,
                            max_length=2000)

    output = ''
    # print('streamer:', streamer)
    for new_output in streamer:
        escapped = chat.escape_markdown(new_output)
        output += escapped
    output = " ".join(output.split()).replace("\n", "")
    # print('output:', output)
    return output


def write_to_json(obj_dict, json_file_path):
    with open(json_file_path, 'a') as json_file:
        json_file.write(json.dumps(obj_dict) + '\n')

def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cudnn.benchmark = False
    cudnn.deterministic = True

    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    device = 'cuda:{}'.format(args.gpu_id)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model = model.eval()

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    chat = Chat(model, vis_processor, device=device)

    # Generate caption based on one image
    # image_path = "./data/examples_v2/cockdial.png"

    with open(args.img_path) as f:
        data_json = []
        for line in f:
            data_json.append(json.loads(line))
    for datas in tqdm(data_json):
        for img_path, data in datas.items():
            for question, gt in data.items():
                prompt = f"You are an expert of emotion understanding. Look at this image, {question}"
                annotation = Generate4img(chat, "/home/czb/project/MiniGPT-4-captions/data/images/" + img_path, prompt)
                write_to_json({img_path: annotation}, args.out_json)


if __name__ == "__main__":
    main()


# screen python ec_complex_minigpt4v2.py --cfg-path eval_configs/minigptv2_eval.yaml  --gpu-id 0 --img-path ec_complex.jsonl --out-json minigpt_complex.jsonl