# EmCoBench: An Extensive Benchmark for General Emotion Comprehension

EmCoBench is a comprehensive benchmark designed for evaluating systems on their ability to understand and identify emotional triggers, rather than just classifying emotions. This is essential for developing more empathetic and human-like AI systems.

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
   ```bash
   python EmCoBench/baselines/qwen/qwen_user.py --model_path Qwen/Qwen-VL-Chat --input_json EmCoBench/EmCo_Basic/user.jsonl --output_json qwen_basic.jsonl --image-path datasets/
   ```
