import gradio as gr
from PIL import Image
import json
import sys

if len(sys.argv) > 1:
    port = int(sys.argv[1])
else:
    port = 7860

data_iter = None
trig_dict = None
now_json_file = "EIBench/EI_Basic/user_sample.jsonl"
process_file = f"web_ann_score/score_process_{port}.json"
save_file = f"web_ann_score/output/score_{port}.jsonl"


def load_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def get_ann(path="web_ann/store.json"):
    with open(path) as f:
        store = json.load(f)
        return store


def switch_label_based_on_dropdown(selected_option):
    # 根据下拉框选项切换标签
    return selected_option


def update_label(score, img_path, index):
    # 获取当前数据
    global data_iter
    current_data = next(data_iter, None)
    index_num = int(index)
    index_num += 1

    # 如果当前数据不为 None，则获取图片路径和标签
    if current_data is not None:
        for image_path, question in current_data.items():
            with open(save_file, "a") as f:
                # 将数据追加到 JSONL 文件中
                json.dump({img_path: score}, f)
                f.write("\n")

            with open(process_file, "w") as f:
                json.dump({now_json_file: index_num}, f)
            # 返回图片和新标签供下一次展示
            global trig_dict
            triggle = trig_dict[image_path]

            return Image.open(image_path), image_path, question, triggle, index_num
    else:
        with open(save_file, "a") as f:
            # 将数据追加到 JSONL 文件中
            json.dump({img_path: score}, f)
            f.write("\n")
        # exit(0)
        # 如果数据迭代完成，则返回 None
        return None, None, None, index


def load_initial_data(i, trig_dict):
    initial_data = next(i, None)
    if initial_data is not None:
        for image_path, question in initial_data.items():
            if image_path.startswith("/home/lyx/datasets"):
                path = image_path
            else:
                path = "/home/lyx/datasets/" + image_path
            triggle = trig_dict[image_path]
            return Image.open(path), path, question, triggle
    else:
        return None, None, None, None


def init():
    trig = get_ann("EIBench/EI_Basic/basic_ground_truth.json")
    global trig_dict
    trig_dict = trig
    process = get_ann(process_file)
    now_json = list(process.keys())
    index = process[now_json[0]]

    print(index)
    data = load_jsonl(now_json[0])
    data = data[index:]
    global data_iter
    data_iter = iter(data)
    img, img_path, question, triggle = load_initial_data(data_iter, trig)
    return img, img_path, question, triggle, index


note = """

**Hello Valuable Volunteers,**

We kindly request your assistance in evaluating the annotation quality of our lab's EIBench dataset. Please rate the quality on a scale from 0 to 5, focusing primarily on whether the "Question" and "Trigger" can satisfactorily answer the query.

**Dataset Annotation Method:**

- If it's your first time visiting the webpage or if you have refreshed the webpage, please click the "Show" button to load the image.
- For subsequent annotations, simply click the "Next" button.
- The types of triggers include but not limited to `Atmosphere; Social Interactions; Body Movements; Facial Expressions; Objects; Performances; Outdoor Activities; Clothing; Sports;`. You may rate them 4-5 if they make sense; consider scoring below 3 only if there are errors or lack some of them.

**Online Data Annotation URL:**

- http://prefix:{port}/
- The task is expected to take approximately 45-75 minutes to complete.

**Thank you for your time and effort!**

MIPS Lab, Shenzhen Technology University

"""

demo = gr.Blocks()
with demo:
    questions = load_jsonl(now_json_file)
    process = get_ann(process_file)
    now_json = list(process.keys())
    index_num = process[now_json[0]]
    total_num = len(questions)
    initial_image = None
    initial_label = None
    image_path = None
    gr.Markdown(note)
    path = ""

    with gr.Row():
        img = gr.Image(value=initial_image, label="Image")
        with gr.Column():
            question = gr.Textbox(
                value=initial_label, label="Question", interactive=False
            )
            triggle = gr.Textbox(
                value=initial_label, label="Triggle", interactive=False
            )
            img_path = gr.Textbox(
                value=image_path,
                label="Image_path",
                interactive=False,
            )
            score_input = gr.Radio(
                choices=["0", "1", "2", "3", "4", "5"],
                label="Rating",
                value="5",  # default value
            )
            with gr.Row():
                index = gr.Textbox(value=index_num, label="index", interactive=False)
                total = gr.Textbox(value=total_num, label="total", interactive=False)
            with gr.Row():
                next_btn = gr.Button("Next")
                show_btn = gr.Button("Show")

    show_btn.click(
        fn=init, inputs=None, outputs=[img, img_path, question, triggle, index]
    )

    next_btn.click(
        fn=update_label,
        inputs=[score_input, img_path, index],
        outputs=[img, img_path, question, triggle, index],
    )


demo.launch(server_name="0.0.0.0", server_port=port)
