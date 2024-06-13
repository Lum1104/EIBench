# We would update the new version of Video-LLaVA code later.
# For Video-LLaVA has refractor their code earlier.
# The following code support the old version codebase for Video-LLaVA inference.
import argparse
import torch

from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria

from tqdm import tqdm
import json


def write_to_json(obj_dict, json_file_path):
    with open(json_file_path, 'a') as json_file:
        json_file.write(json.dumps(obj_dict) + '\n')


def main(args):
    # Model
    disable_torch_init()
    assert not (args.image_file and args.video_file)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name,
                                                                     args.load_8bit, args.load_4bit, device=args.device)
    # print(model, tokenizer, processor)
    image_processor = processor['image']
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode
    
    # user.jsonl
    # for basic emotion comprehension
    with open(args.image_file) as f:
        data_json = []
        for line in f:
            data_json.append(json.loads(line))

    # ec_complex.jsonl
    # for complex emotion comprehension
    # for datas in tqdm(data_json):
    #     for img_path, data in datas.items():
    #         for question, gt in data.items():

    for data in tqdm(data_json):
        for img_path, question in data.items():
            conv = conv_templates[args.conv_mode].copy()
            if "mpt" in model_name.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles
            image_tensor = image_processor.preprocess(img_path, return_tensors='pt')['pixel_values']
            if type(image_tensor) is list:
                tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                tensor = image_tensor.to(model.device, dtype=torch.float16)
            key = ['image']
            # inp = f"You are a good expert of emotion understanding. Look at the image, {question}"
            inp = f"You are a good expert of emotion understanding. Look at the image, {question} Let's think step by step."
            inp = DEFAULT_X_TOKEN['IMAGE'] + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['IMAGE'], return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[tensor, key],
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace(' </s>', '').replace('<s> ', '').replace("<|im_end|>", "").strip()

            # video-llava.jsonl
            write_to_json({img_path: outputs}, args.out_json)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None, required=True)
    parser.add_argument("--out-json", type=str, default=None, required=True)
    parser.add_argument("--video-file", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)
