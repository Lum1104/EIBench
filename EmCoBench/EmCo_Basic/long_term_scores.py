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