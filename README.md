# EmCoBench: An Extensive Benchmark for General Emotion Comprehension

EmCoBench is a comprehensive benchmark designed for evaluating systems on their ability to understand and identify emotional triggers, rather than just classifying emotions. This is essential for developing more empathetic and human-like AI systems.

***More details about EmCoBench, including a forthcoming paper, will be available soon.***

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
```

```python
import spacy
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to("cuda")

def get_sentence_embeddings(sentences):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings

def calculate_coherence(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    if len(sentences) < 2:
        return None
    embeddings = get_sentence_embeddings(sentences)
    
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        similarities.append(sim)
    if not similarities:
        return None
    coherence_score = np.mean(similarities)
    return coherence_score


with open("path/to/ec_data.jsonl", 'r') as f:
    ec_data_jsons = []
    for line in f:
        ec_data_jsons.append(json.loads(line))


coherence_scores = []

for ec_data_json in tqdm(ec_data_jsons):
    for img_path, data_input in ec_data_json.items():
        coherence_score = calculate_coherence(data_input)
        if coherence_score:
            coherence_scores.append(coherence_score)

print(sum(coherence_scores)/len(coherence_scores))
```
