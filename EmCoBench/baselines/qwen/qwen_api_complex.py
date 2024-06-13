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
    try:
        return response.output.choices[0].message.content[0]["text"]
    except IndexError:
        return "I don't know"
    except:
        return response.output.choices[0].message.content[1]["text"]

if __name__ == '__main__':
    dashscope.api_key = "YOUR_API_KEY"
    with open("ec_complex.jsonl") as f:
        data_json = []
        for line in f:
            data_json.append(json.loads(line))

    for datas in tqdm(data_json):
        for img_path, data in datas.items():
            for question, gt in data.items():
                content = f"You are an expert of emotion understanding. Look at this image, {question.lower()}."
                while True:
                    try:
                        completion = get_response("images/" + img_path, content)
                        write_to_json({img_path: completion}, "api_qwen_complex.jsonl")
                        break
                    except Exception as e:
                        print(e)
                        continue
