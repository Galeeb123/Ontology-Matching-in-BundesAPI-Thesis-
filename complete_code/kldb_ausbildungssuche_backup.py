from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from kldb_input import kldb_occupations, occupations_fur_berufsausbildung


def ausbildungssuche_imp_data(df_ausbildungssuche):
    df_ausbildungssuche_imp = df_ausbildungssuche[['angebot.titel']]
    df_ausbildungssuche_imp.drop_duplicates(inplace=True)
    return df_ausbildungssuche_imp



def encoding_berufenet(model):
    # Multilingual model (works well for German occupation titles)
    df_unique_bndata = occupations_fur_berufsausbildung()
    berufenet_titles = df_unique_bndata["Berufsbenennungen"].tolist()
    berufenet_ids = df_unique_bndata["KldB-2010_(5-Steller)"].tolist()
    berufenet_embeddings = model.encode(
        berufenet_titles,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # cosine-friendly
        )
    return berufenet_titles, berufenet_ids, berufenet_embeddings

def matches_ausbildungssuche_berufenet_SBERT(model, df_ausbildungssuche):
    results = []
    print("Encoding Ausbildung titles and matching to kldb Occupations...")
    df_ausbildungssuche_imp = ausbildungssuche_imp_data(df_ausbildungssuche)
    berufenet_titles, berufenet_ids, berufenet_embeddings = encoding_berufenet(model)
    for idx, row in tqdm(df_ausbildungssuche_imp.iterrows(), total=len(df_ausbildungssuche_imp)):
        aus_title = row["angebot.titel"]
        # Encode this Ausbildung title
        aus_emb = model.encode(
            aus_title,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).reshape(1, -1)

        # Cosine similarity with ALL BERUFENET embeddings
        sims = cosine_similarity(aus_emb, berufenet_embeddings)[0]

        # Get best match
        best_idx = sims.argmax()
        best_score = sims[best_idx]

        results.append({
            "ausbildung_title": aus_title,
            "best_kldb_occupation": berufenet_titles[best_idx],
            "similarity": float(best_score),
            "Kldb_code": berufenet_ids[best_idx] 
        })

    df_sbert_matches = pd.DataFrame(results)
    return df_sbert_matches


if __name__ == "__main__":
    df_ausbildungssuche = pd.read_csv('/home/gshaik@forschungsnetz.local/project/data/raw_data/Ausbildungssuche_complete_data.csv')
    df_ausbildungssuche = pd.read_csv(path_raw_data_ausbildungssuche)
    model = SentenceTransformer("Sahajtomar/German-semantic")
    df_matches = matches_ausbildungssuche_berufenet_SBERT(model, df_ausbildungssuche)
    df_ausbildungssuche_kldb = pd.merge(df_ausbildungssuche, df_matches, left_on="angebot.titel",right_on="ausbildung_title",how="left")
    df_ausbildungssuche_kldb.to_csv('/home/gshaik@forschungsnetz.local/project/data/output/ausbildungssuche_kldb_assigned.csv', index=False)