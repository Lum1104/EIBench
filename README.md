# 🌟 Why We Feel: Breaking Boundaries in Emotional Reasoning with Multimodal Large Language Models

<p align="center">
  <a href="https://arxiv.org/abs/2504.07521">
    <img src="https://img.shields.io/badge/arXiv-2504.07521-b31b1b.svg?logo=arxiv&logoColor=white" alt="arXiv badge"/>
  </a>
</p>

![eibench](https://github.com/Lum1104/EmCoBench/assets/87774050/71870702-477b-49cd-9be1-a8d6a1180e78)

This paper introduces EIBench, a benchmark for evaluating the ability of Vision-Language Models (VLLMs) in the task of Emotion Interpretation (EI). Unlike traditional emotion analysis that focuses primarily on recognizing which emotion is present, EI emphasizes understanding the underlying causes of emotions, including both explicit factors (such as visible objects and interpersonal interactions) and implicit factors (such as cultural context and off-screen events).

## 🔍 Key Highlights

- **Emotion Interpretation Task:** The goal of EI is to explain why an individual experiences a particular emotional response, rather than merely labeling the emotion category. This task requires models to engage in causal reasoning instead of simple emotion classification.
- **Rich Dataset:** Comprising 1,615 basic EI samples and 50 complex EI samples, EIBench covers four primary emotion categories (anger, sadness, excitement, happiness) and complex scenarios with interwoven emotions. Each sample demands rationale-based explanations from models, rather than straightforward categorization.
- **Coarse-to-Fine Self-Ask (CFSA) Annotation Method:** By employing iterative question-and-answer rounds, CFSA guides VLLMs to progressively delve into emotional triggers, generating high-quality annotations that capture both explicit and implicit factors.
- **Comprehensive Model Evaluation:** Under four distinct experimental settings (including using image captions, chain-of-thought prompting, and persona-based variations), both open-source and proprietary large language models were systematically assessed. The results reveal significant performance gaps, especially in complex emotional reasoning scenarios, even for state-of-the-art models like Claude-3 and ChatGPT-4.

## 📦 Prerequisites

To get started with EIBench, you'll need to download and prepare the following datasets:

- [EmoSet-118K](https://vcc.tech/EmoSet)
- [CAER-S](https://caer-dataset.github.io/)

After downloading, unzip these datasets and place them in the datasets folder in your project directory.

## 🛠️ Setup & Usage

To use the EIBench dataset and benchmark in your project:

1. Clone this repository:
```bash
git clone https://github.com/Lum1104/EIBench.git
```

2. Navigate to the directory:
```bash
cd EIBench
```

3. Run the example baseline code and test your own models.

For each baseline model, please install the required environment as needed:
```bash
# Basic EIBench
python EIBench/baselines/qwen/qwen_user.py --model-path Qwen/Qwen-VL-Chat --input-json EIBench/EI_Basic/user.jsonl --output-json EIBench/EI_Basic/qwen_basic.jsonl --image-path datasets/
# Complex EIBench
python EIBench/baselines/qwen/qwen_complex.py --model-path Qwen/Qwen-VL-Chat --input-json EI_Complex/ei_complex.jsonl --output-json EIBench/EI_Complex/qwen_complex.jsonl --image-path datasets/
```
4. Get evaluate results by LLaMA-3/ChatGPT-3.5

Here is the script for LLaMA-3 evaluation.
```bash
# Basic EIBench
cd EIBench/EI_Basic/
python llama3-eval.py --model-id meta-llama/Meta-Llama-3-8B-Instruct --ec-data-file qwen_basic.jsonl --gt-file basic_ground_truth.json --output-file qwen_basic_scores_llama3.jsonl
python get_scores.py --file-path qwen_basic_llama3_scores.jsonl
# Complex EIBench
cd EIBench/EI_Complex/
python llama3-eval-complex.py --ec-data-file qwen_complex.jsonl --gt-file ei_complex.jsonl --output-file qwen_complex_llama3_scores.jsonl --model-id meta-llama/Meta-Llama-3-8B-Instruct
```

Here is the script for ChatGPT-3.5 evaluation. Prepare your api key and write it in the variable `OpenAI(api_key="YOUR_API_KEY")`.
```bash
# Basic EIBench
cd EIBench/EI_Basic/
python gpt-eval.py --ec-data-file qwen_basic.jsonl --gt-file basic_ground_truth.json --output-file qwen_basic_scores_gpt.jsonl
python get_scores.py --file-path qwen_basic_gpt_scores.jsonl
# Complex EIBench
cd EIBench/EI_Complex/
python gpt-eval-complex.py --ec-data-file qwen_complex.jsonl --gt-file ei_complex.jsonl --output-file qwen_complex_gpt_scores.jsonl
```

We also provide evaluation code for Long-term Coherence. Please install the required packages:
```bash
pip install spacy
pip -m spacy download en_core_web_sm
cd EIBench/EI_Basic/
python long_term_scores.py --file-path path/to/ei_data.jsonl
```

## Baselines
### Close-source Models
```bash
# (gpt4o/gpt4v)
python gpt4-basic.py --ec-data-file path/to/user.jsonl --image-path path/to/dataset/ --output-file gpt4o_user.jsonl
python gpt4-score-complex.py --gt-file path/to/ei_complex.jsonl --image-path path/to/dataset/ --output-file gpt4o_complex.jsonl
# (Claude-3-haiku/Claude-3-sonnet)
python claude_basic.py --ec-data-file path/to/user.jsonl --image-path path/to/dataset/ --output-file claude_haiku_user.jsonl
python claude_complex.py --gt-file path/to/ei_complex.jsonl --image-path path/to/dataset/ --output-file claude_haiku_complex.jsonl
# qwen-vl-plus
python qwen_api_basic.py --ec-data-file path/to/user.jsonl --image-path path/to/datasets/ --output-file qwen_api_user.jsonl
python qwen_api_complex.py --gt-file path/to/ei_complex.jsonl --image-path path/to/dataset --output-file qwen_qpi_complex.jsonl
```
### Open-source Models
Please follow the environment needed by each baseline models:
#### LLaVA
```bash
cd LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
# Input different LLaVA model to get the evaluation results.
python -m llava.serve.ei_basic_llava --model-path liuhaotian/llava-v1.6-34b --image-file path/to/user.jsonl --out-json llava34b_user.jsonl --image-path path/to/dataset/
python -m llava.serve.ei_complex_llava --model-path liuhaotian/llava-v1.6-34b --image-file path/to/ei_complex.jsonl --out-json llava34b_complex.jsonl --image-path path/to/dataset/
```
#### MiniGPT4-v2
```bash
cd MiniGPT4-v2
conda env create -f environment.yml
conda activate minigptv
# Modify MiniGPT4-v2/eval_configs/minigptv2_eval.yaml
python ei_basic_minigpt4v2.py --cfg-path eval_configs/minigptv2_eval.yaml  --gpu-id 0 --img-path path/to/user.jsonl --out-json minigpt4v2_user.jsonl --dataset-path path/to/dataset/
python ei_complex_minigpt4v2.py --cfg-path eval_configs/minigptv2_eval.yaml  --gpu-id 0 --img-path path/to/ei_complex.jsonl --out-json minigpt_complex.jsonl --dataset-path path/to/dataset/
```
#### Otter
```bash
cd Otter
conda env create -f environment.yml
conda activate otter
python ei_basic_otter.py --ec-data-file path/to/user.jsonl --image-path path/to/datasets/ --output-file otter_user.jsonl
python ei_complex_otter.py --gt-file path/to/ei_complex.jsonl --image-path path/to/dataset/ --output-file otter_complex.jsonl
```

Feel free to explore, contribute, and raise issues if you run into any trouble!
