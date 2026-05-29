from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

from kldb_input import kldb_occupations



'''
Ontology Matching Methods Used
	•	Fuzzy String Matching
A character-based lexical method that measures similarity based on string overlap and handles minor spelling variations but does not capture semantic meaning.
	•	TF-IDF with Cosine Similarity
A lexical vector-space approach that represents text using term-frequency weights and compares texts based on weighted word overlap.
	•	FastText Embeddings
A subword-based embedding method that captures basic semantic similarity and handles morphological variations and compound words.
	•	BERT-base Embeddings
A transformer-based language model that produces contextual embeddings; sentence representations are obtained via pooling and used as a transformer baseline for similarity matching.
	•	Sentence-BERT (SBERT)
A transformer model explicitly optimized for semantic similarity using a siamese architecture, providing high-quality sentence embeddings and state-of-the-art performance for ontology matching.

Below only State-of-the-ART, Sentance-BERT is considered
'''

def bewerberboerse_imp_data(df_bewerberboerse):
    df_bewerberboerse_imp = df_bewerberboerse[['berufe']]
    df_bewerberboerse_imp.drop_duplicates(inplace=True)
    return df_bewerberboerse_imp



def encoding_berufenet(model):
    # Multilingual model (works well for German occupation titles)
    df_unique_bndata = kldb_occupations()
    berufenet_titles = df_unique_bndata["Berufsbenennungen"].tolist()
    berufenet_ids = df_unique_bndata["KldB-2010_(5-Steller)"].tolist()
    berufenet_embeddings = model.encode(
        berufenet_titles,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # cosine-friendly
        )
    return berufenet_titles, berufenet_ids, berufenet_embeddings

def matches_bewerberboerse_berufenet_SBERT(model, df_bewerberboerse):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")
    
    df_bewerberboerse_imp = bewerberboerse_imp_data(df_bewerberboerse)
    job_titles = df_bewerberboerse_imp["berufe"].tolist()

    # Encode BERUFENET once
    berufenet_titles, berufenet_ids, berufenet_embeddings = encoding_berufenet(model)

    results = []

    batch_size = 512  # adjust based on memory

    print("Processing in batches...")

    for i in tqdm(range(0, len(job_titles), batch_size)):
        batch_titles = job_titles[i:i+batch_size]

        # Encode batch
        batch_emb = model.encode(
            batch_titles,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device
        )

        # Compute similarity ONLY for batch
        sims = np.dot(batch_emb, berufenet_embeddings.T)

        # Get best matches
        best_idx = sims.argmax(axis=1)
        best_scores = sims[np.arange(len(batch_titles)), best_idx]

        for j in range(len(batch_titles)):
            results.append({
                "job_title": batch_titles[j],
                "best_kldb_occupation": berufenet_titles[best_idx[j]],
                "similarity": float(best_scores[j]),
                "Kldb_code": berufenet_ids[best_idx[j]]
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print('Ontology matching on bewerberboerse ....')
    df_bewerberboerse = pd.read_csv('/home/galeebs/project/data_2/raw_data/bewerberboerse_complete_data.csv')
    df_bewerberboerse.drop_duplicates(subset="refnr", inplace=True) # Removing duplicates based on 'refnr'
    model = SentenceTransformer("Sahajtomar/German-semantic", device=device)
    df_matches = matches_bewerberboerse_berufenet_SBERT(model, df_bewerberboerse)
    df_bewerberboerse_kldb = pd.merge(df_bewerberboerse, df_matches, left_on="berufe",right_on="job_title",how="left")
    df_bewerberboerse_kldb.to_csv('/home/galeebs/project/data_2/output/bewerberboerse_kldb_assigned.csv', index=False)