from http import HTTPStatus
from dashscope import MultiModalConversation
import dashscope
import base64
import json
from tqdm import tqdm

def write_to_json(obj_dict, json_file_path):
    with open(json_file_path, 'a') as json_file:
        json_file.write(json.dumps(obj_dict) + '\n')

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_response(img_path, content):
    print(img_path, content)
    messages = [{
        'role': 'user',
        'content': [
            {
                'image': "file://" + img_path
            },
            {
                'text': content
            },
        ]
    }]
    response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
    print(response)
    return response.output.choices[0].message.content[0]["text"]

if __name__ == '__main__':
    dashscope.api_key = "YOUR_API_KEY"
    with open("user.jsonl", 'r') as f:
        ec_data = []
        for line in f:
            ec_data.append(json.loads(line))
    
    for data in tqdm(ec_data):
        for img_path, data_input in data.items():
            content = f"You are an expert of emotion understanding. Look at this image, {data_input.lower()}"
            try:
                completion = get_response("path/to/dataset/"+img_path, content)
                write_to_json({img_path: completion}, "api_qwen.jsonl")
            except:
                continue