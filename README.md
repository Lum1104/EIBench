# EmCoBench: An Extensive Benchmark for General Emotion Comprehension

![emcobench](https://github.com/Lum1104/EmCoBench/assets/87774050/71870702-477b-49cd-9be1-a8d6a1180e78)

EmCoBench is a comprehensive benchmark designed for evaluating systems on their ability to understand and identify emotional triggers, rather than just classifying emotions. This is essential for developing more empathetic and human-like AI systems.

## Authors

[Yuxiang Lin](https://lum1104.github.io)<sup>1</sup>, [Jue Wang](https://a-new-b.github.io/)<sup>1,4</sup>, Haomin Liang<sup>1</sup>, Zebang Cheng<sup>1</sup>, [Jun-Yan He](https://scholar.google.com/citations?hl=en&user=bjNZqGAAAAAJ&view_op=list_works&authuser=1&sortby=pubdate)<sup>2*</sup>, [Zhi-Qi Cheng](https://zhiqic.github.io/homepage/index.html)<sup>3*</sup>, [Xiaojiang Peng](https://pengxj.github.io/)<sup>1*</sup>, [Alexander G. Hauptmann](https://www.cs.cmu.edu/~./alex/)<sup>3</sup> [\*Corresponding Authors]

<sup>1</sup>[Shenzhen Technology University](https://www.sztu.edu.cn/), <sup>2</sup>Alibaba Group, <sup>3</sup>[Carnegie Mellon University](https://www.cs.cmu.edu/), <sup>4</sup>[Shenzhen Institute of Advanced Technology](https://english.siat.ac.cn/)

***More details about EmCoBench, please refer to this [report](https://lum1104.github.io/resources/emcobench_paper.pdf). Feel free to email yuxiang.lin@gatech.edu if you have any question.***

## Overview

- **Emotion Comprehension Task**: Focuses on identifying emotional triggers, providing a deeper understanding of emotions.
- **EmCoBench Dataset**: Includes 78 fine-grained emotions and 1,655 emotion comprehension samples, with 50 multifaceted complex samples.

## Prerequisites

Download the following datasets before using EmCoBench:
- [EmoSet-118K](https://vcc.tech/EmoSet)
- [CAER-S](https://caer-dataset.github.io/)

Unzip and place them in the `dataset` folder.

## Usage

To use the EmCoBench dataset and benchmark in your project:

1. Clone this repository:
```bash
git clone https://github.com/Lum1104/EmCoBench.git
```

2. Navigate to the directory:
```bash
cd EmCoBench
```

3. Run the example baseline code and test your own models.

For each baseline model, please install the required environment as needed:
```bash
# Basic EmCoBench
python EmCoBench/baselines/qwen/qwen_user.py --model-path Qwen/Qwen-VL-Chat --input-json EmCoBench/EmCo_Basic/user.jsonl --output-json EmCoBench/EmCo_Basic/qwen_basic.jsonl --image-path datasets/
# Complex EmCoBench
python EmCoBench/baselines/qwen/qwen_complex.py --model-path Qwen/Qwen-VL-Chat --input-json EmCo_Complex/ec_complex.jsonl --output-json EmCoBench/EmCo_Complex/qwen_complex.jsonl --image-path datasets/
```
4. Get evaluate results by LLaMA-3/ChatGPT-3.5

Here is the script for LLaMA-3 evaluation.
```bash
# Basic EmCoBench
cd EmCoBench/EmCo_Basic/
python llama3-eval.py --model-id meta-llama/Meta-Llama-3-8B-Instruct --ec-data-file qwen_basic.jsonl --gt-file basic_ground_truth.json --output-file qwen_basic_scores_llama3.jsonl
python get_scores.py --file-path qwen_basic_llama3_scores.jsonl
# Complex EmCoBench
cd EmCoBench/EmCo_Complex/
python llama3-eval-complex.py --ec-data-file qwen_complex.jsonl --gt-file ec_complex.jsonl --output-file qwen_complex_llama3_scores.jsonl --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

Here is the script for ChatGPT-3.5 evaluation. Prepare your api key and write it in the variable `OpenAI(api_key="YOUR_API_KEY")`.
```bash
# Basic EmCoBench
cd EmCoBench/EmCo_Basic/
python gpt-eval.py --ec-data-file qwen_basic.jsonl --gt-file basic_ground_truth.json --output-file qwen_basic_scores_gpt.jsonl
python get_scores.py --file-path qwen_basic_gpt_scores.jsonl
# Complex EmCoBench
cd EmCoBench/EmCo_Complex/
python gpt-eval-complex.py --ec-data-file qwen_complex.jsonl --gt-file ec_complex.jsonl --output-file qwen_complex_gpt_scores.jsonl
```

We also provide evaluation code for Long-term Coherence. Please install the required packages:
```bash
pip install spacy
pip -m spacy download en_core_web_sm
cd EmCoBench/EmCo_Basic/
python long_term_scores.py --file-path path/to/ec_data.jsonl
```

## Baselines
### Close-source Models
```bash
# (gpt4o/gpt4v)
python gpt4-basic.py --ec-data-file path/to/user.jsonl --image-path path/to/dataset/ --output-file gpt4o_user.jsonl
python gpt4-score-complex.py --gt-file path/to/ec_complex.jsonl --image-path path/to/dataset/ --output-file gpt4o_complex.jsonl
# (Claude-3-haiku/Claude-3-sonnet)
python claude_basic.py --ec-data-file path/to/user.jsonl --image-path path/to/dataset/ --output-file claude_haiku_user.jsonl
python claude_complex.py --gt-file path/to/ec_complex.jsonl --image-path path/to/dataset/ --output-file claude_haiku_complex.jsonl
# qwen-vl-plus
python qwen_api_basic.py --ec-data-file path/to/user.jsonl --image-path path/to/datasets/ --output-file qwen_api_user.jsonl
python qwen_api_complex.py --gt-file path/to/ec_complex.jsonl --image-path path/to/dataset --output-file qwen_qpi_complex.jsonl
```
### Open-source Models
Please follow the enviornment needed by each baseline models:
#### LLaVA
```bash
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
# Input different LLaVA model to get the evaluation results.
python -m llava.serve.ec_basic_llava --model-path liuhaotian/llava-v1.6-34b --image-file path/to/user.jsonl --out-json llava34b_user.jsonl --image-path path/to/dataset/
python -m llava.serve.ec_complex_llava --model-path liuhaotian/llava-v1.6-34b --image-file path/to/ec_complex.jsonl --out-json llava34b_complex.jsonl --image-path path/to/dataset/
```
#### MiniGPT4-v2
```bash
cd MiniGPT4-v2
conda env create -f environment.yml
conda activate minigptv
# Modify MiniGPT4-v2/eval_configs/minigptv2_eval.yaml
python ec_basic_minigpt4v2.py --cfg-path eval_configs/minigptv2_eval.yaml  --gpu-id 0 --img-path path/to/user.jsonl --out-json minigpt4v2_user.jsonl --dataset-path path/to/dataset/
python ec_complex_minigpt4v2.py --cfg-path eval_configs/minigptv2_eval.yaml  --gpu-id 0 --img-path path/to/ec_complex.jsonl --out-json minigpt_complex.jsonl --dataset-path path/to/dataset/
```
#### Otter
```bash
cd Otter
conda env create -f environment.yml
conda activate otter
python ec_basic_otter.py --ec-data-file path/to/user.jsonl --image-path path/to/datasets/ --output-file otter_user.jsonl
python ec_complex_otter.py --gt-file path/to/ec_complex.jsonl --image-path path/to/dataset/ --output-file otter_complex.jsonl
```
