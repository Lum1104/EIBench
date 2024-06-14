import spacy
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from tqdm import tqdm
import argparse

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the BERT tokenizer and model
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

# Set up argparse to handle the input file path
parser = argparse.ArgumentParser(description='Calculate coherence scores for text data.')
parser.add_argument('--file-path', type=str, help='Path to the ec_data.jsonl file')
args = parser.parse_args()

# Read the input file
with open(args.file_path, 'r') as f:
    ec_data_jsons = []
    for line in f:
        ec_data_jsons.append(json.loads(line))

# Calculate coherence scores
coherence_scores = []

for ec_data_json in tqdm(ec_data_jsons):
    for img_path, data_input in ec_data_json.items():
        coherence_score = calculate_coherence(data_input)
        if coherence_score:
            coherence_scores.append(coherence_score)

# Print the average coherence score
if coherence_scores:
    print(f'Average Coherence Score: {sum(coherence_scores) / len(coherence_scores)}')
else:
    print('No valid coherence scores were calculated.')
