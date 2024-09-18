# EmCoBench: An Extensive Benchmark for General Emotion Comprehension

![emcobench](https://github.com/Lum1104/EmCoBench/assets/87774050/71870702-477b-49cd-9be1-a8d6a1180e78)

EmCoBench is a comprehensive benchmark designed for evaluating systems on their ability to understand and identify emotional triggers, rather than just classifying emotions. This is essential for developing more empathetic and human-like AI systems.

## Overview

- **Emotion Comprehension Task**: Focuses on identifying emotional triggers, providing a deeper understanding of emotions.
- **EmCoBench Dataset**: Includes 78 fine-grained emotions and 1,655 emotion comprehension samples, with 50 multifaceted complex samples.

## Benchmark

### Basic Emotion Comprehension Performance of Open-Source/Close-Source Language Models

| Models                | Happy       | Angry       | Sadness     | Excitement  | Overall     |
|-----------------------|-------------|-------------|-------------|-------------|-------------|
| **_User Question_**   |             |             |             |             |             |
| Qwen-VL-Chat          | 32.09/39.68 | 22.32/26.10 | 30.64/33.88 | 25.02/36.32 | 26.45/33.65 |
| Video-LLaVA           | 55.55/53.28 | 40.42/36.97 | 50.62/45.25 | 51.78/52.23 | 49.26/47.06 |
| MiniGPT-v2            | 52.78/51.80 | **47.10/47.76** | **60.47/58.14** | 50.78/53.66 | 52.89/53.59 |
| Otter                 | 45.63/49.25 | 42.53/43.07 | 47.67/46.19 | 39.47/48.30 | 42.81/46.64 |
| LLaVA-1.5 (13B)       | **59.01/57.52** | 45.44/41.88 | 55.16/48.64 | **57.46/58.73** | **54.37/52.20** |
| LLaVA-NEXT (7B)       | 54.16/49.24 | 43.71/39.87 | 53.29/46.52 | 58.90/53.06 | 53.82/48.18 |
| LLaVA-NEXT (13B)      | 57.17/55.18 | 43.16/37.93 | 54.16/45.42 | 59.38/55.29 | 54.33/48.79 |
| LLaVA-NEXT (34B)      | 54.50/51.03 | 38.96/35.65 | 51.10/47.21 | 51.77/52.04 | 49.03/47.13 |
| **_User Question & Caption_** |             |             |             |             |             |
| Qwen-VL-Chat          | 41.94/46.34 | 32.71/31.91 | 41.82/44.16 | 38.65/43.84 | 38.47/41.54 |
| Video-LLaVA           | 56.77/58.79 | 43.65/43.86 | 54.25/55.12 | 55.35/59.42 | 52.63/54.85 |
| MiniGPT-v2            | 55.11/60.04 | 47.95/51.00 | **62.29/64.24** | 51.55/57.90 | 54.05/58.37 |
| Otter                 | 48.97/54.67 | 34.22/37.12 | 34.57/37.55 | 35.27/42.99 | 35.62/40.85 |
| LLaVA-1.5 (13B)       | 57.91/58.46 | 43.75/40.72 | 55.47/51.46 | 56.42/59.42 | 53.55/53.13 |
| LLaVA-NEXT (7B)       | **64.32/61.00** | 48.60/46.74 | 58.75/53.00 | **62.99/59.39** | 58.80/54.97 |
| LLaVA-NEXT (13B)      | 61.99/61.95 | **48.84/46.85** | 59.62/55.18 | 62.17/59.95 | **58.60/55.92** |
| LLaVA-NEXT (34B)      | 57.51/62.73 | 46.47/47.87 | 58.35/55.84 | 60.17/59.64 | 56.60/56.24 |
| LLaMA-3 (8B) (Text Only) | 52.36/50.73 | 34.78/32.71 | 52.29/46.87 | 43.62/42.06 | 44.73/41.94 |
| **_User Question & CoT_** |             |             |             |             |             |
| Qwen-VL-Chat          | 41.99/44.46 | 34.62/31.06 | 43.64/39.30 | 32.78/40.04 | 36.79/38.18 |
| Video-LLaVA           | 51.42/47.63 | 42.68/35.65 | 56.77/46.29 | 53.01/46.98 | 51.81/44.42 |
| MiniGPT-v2            | 56.36/57.58 | 47.71/48.32 | **59.46/56.79** | 50.21/52.39 | 52.67/53.08 |
| Otter                 | 49.97/51.91 | 43.23/43.71 | 50.15/46.86 | 42.30/47.16 | 45.17/46.61 |
| LLaVA-1.5 (13B)       | **59.12/56.94** | 40.97/34.44 | 53.07/45.66 | 54.16/54.36 | 51.34/47.80 |
| LLaVA-NEXT (7B)       | 54.74/52.04 | 44.61/41.93 | 52.69/47.63 | 52.78/47.60 | 51.14/46.66 |
| LLaVA-NEXT (13B)      | 50.91/50.35 | 42.21/38.81 | 54.66/49.42 | 51.64/49.39 | 50.47/47.21 |
| LLaVA-NEXT (34B)      | 52.17/49.55 | **48.35/44.45** | 55.97/50.55 | **55.29/53.46** | **53.84/50.50** |
| CFSA (LLaVA-NEXT (34B)) | 69.68/68.72 | 61.08/61.14 | 68.39/69.46 | 72.63/70.31 | 68.81/68.04 |
| **_Close-source Models_** |             |             |             |             |             |
| Qwen-vl-plus          | 29.05/27.22 | 23.58/17.89 | 38.35/30.08 | 30.09/26.87 | 31.00/25.90 |
| ChatGPT-4V            | 52.30/55.74 | 48.93/48.57 | 45.00/44.42 | 46.38/49.90 | 46.86/48.58 |
| ChatGPT-4o            | 52.94/50.78 | 42.12/35.33 | 49.79/46.42 | 53.48/54.53 | 49.99/47.93 |
| Claude-3-haiku        | **59.20/60.28** | **49.87/49.84** | **67.21/63.26** | **67.55/68.10** | **63.24/62.41** |
| Claude-3-sonnet       | 44.58/44.45 | 38.95/42.86 | 55.98/54.40 | 61.41/62.24 | 54.10/54.89 |

### Multi-faceted Emotion Comprehension Performance of Open-Source/Close-Source Language Models

| **Models**            | **Recall**       |
|-----------------------|------------------|
| **_Open-Source_**     |                  |
| Qwen-VL-Chat          | 22.00/32.40      |
| Video-LLaVA           | 30.90/32.27      |
| MiniGPT-v2            | 35.10/36.00      |
| Otter                 | 27.90/33.23      |
| LLaVA-1.5 (13B)       | _38.10/39.53_    |
| LLaVA-NEXT (7B)       | 38.71/33.50      |
| LLaVA-NEXT (13B)      | 39.16/33.60      |
| LLaVA-NEXT (34B)      | 35.37/33.10      |
| **_Close-Source_**    |                  |
| Qwen-vl-plus          | 20.37/19.60      |
| Claude-3-haiku        | 24.00/24.77      |
| Claude-3-sonnet       | 21.37/22.40      |
| ChatGPT-4V            | 28.00/30.60      |
| ChatGPT-4o            | **39.27/39.57**  |

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
