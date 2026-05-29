from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

from kldb_input import kldb_occupations


def preprocess_text(text):
    """
    Lightweight preprocessing for German occupational data
    """

    text = str(text).lower()

    # -----------------------------------------
    # Remove recruitment markers
    # -----------------------------------------

    text = re.sub(r'\(m/w/d\)', ' ', text)
    text = re.sub(r'\(m/w/x\)', ' ', text)
    text = re.sub(r'\(gn\)', ' ', text)

    # -----------------------------------------
    # Remove punctuation
    # -----------------------------------------

    text = re.sub(r'[^\w\s]', ' ', text)

    # -----------------------------------------
    # Remove extra spaces
    # -----------------------------------------

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def bewerberboerse_imp_data(df_bewerberboerse):
    df_bewerberboerse_imp = df_bewerberboerse[['berufe_clean']]
    df_bewerberboerse_imp.drop_duplicates(inplace=True)
    return df_bewerberboerse_imp



def encoding_berufenet(model):
    # Multilingual model (works well for German occupation titles)
    df_unique_bndata = kldb_occupations()
    df_unique_bndata["Berufsbenennungen_clean"] = df_unique_bndata["Berufsbenennungen"].astype(str).apply(preprocess_text)
    berufenet_titles = df_unique_bndata["Berufsbenennungen_clean"].tolist()
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
    job_titles = df_bewerberboerse_imp["berufe_clean"].tolist()

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
    df_bewerberboerse['berufe_clean'] = df_bewerberboerse["berufe"].astype(str).apply(preprocess_text)
    model = SentenceTransformer("Sahajtomar/German-semantic", device=device)
    df_matches = matches_bewerberboerse_berufenet_SBERT(model, df_bewerberboerse)
    df_bewerberboerse_kldb = pd.merge(df_bewerberboerse, df_matches, left_on="berufe_clean",right_on="job_title",how="left")
    df_bewerberboerse_kldb.drop("berufe_clean", axis=1, inplace=True)
    df_bewerberboerse_kldb.to_csv('/home/galeebs/project/data_2/output/bewerberboerse_kldb_assigned.csv', index=False)