import os
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

def sentence_similarity(sentence1, sentence2, model_name="BAAI/bge-m3"):
    # Load the model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Encode sentences
    encoded_sentence1 = tokenizer(sentence1, return_tensors="pt")
    encoded_sentence2 = tokenizer(sentence2, return_tensors="pt")

    # Get embeddings
    with torch.no_grad():
        output1 = model(**encoded_sentence1)
        output2 = model(**encoded_sentence2)

    # Extract embeddings from the [CLS] token (index 0)
    embedding1 = output1.last_hidden_state[:, 0, :]
    embedding2 = output2.last_hidden_state[:, 0, :]

    # Calculate cosine similarity
    similarity_score = F.cosine_similarity(embedding1, embedding2).item()
    
    return similarity_score

# sentence1 = "This is a sample sentence."
# sentence2 = "Here is an example sentence."
# similarity = sentence_similarity(sentence1, sentence2)

# print(f"Similarity between the sentences: {similarity}")
