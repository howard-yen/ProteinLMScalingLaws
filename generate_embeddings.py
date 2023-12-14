import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="csv file path with pairs protein name and protein sequence")
    parser.add_argument("--output_path", type=str, default="output path for embeddings")
    parser.add_argument("--model_path", type=str, default="path to model")

    args = parser.parse_args()
    data = pd.read_csv(args.data_path, header=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path).to(device)

    embeddings = []
    with torch.inference_mode():
        for index, row in tqdm(data.iterrows(), total=data.shape[0]):
            inputs = tokenizer(row[1], return_tensors="pt", max_length=1024, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden_states = outputs[0]
            embedding = mean_pooling(last_hidden_states, inputs['attention_mask'])
            embeddings.append([row[0]] + embedding.tolist()[0])
            inputs = {k: v.cpu() for k, v in inputs.items()}
    embeddings.insert(0, ["PDB_ID" if "SKEMPI" in args.data_path else "Entry"] + [i for i in range(embedding.shape[1])])
    embeddings = pd.DataFrame(embeddings)
    embeddings.to_csv(args.output_path, index=False, header=None)

if __name__ == "__main__":
    main()
