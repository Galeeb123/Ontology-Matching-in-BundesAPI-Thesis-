from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
import re

from kldb_input import (
    kldb_occupations,
    occupations_fur_berufsausbildung
)

# =====================================================
# GPU CONFIGURATION
# =====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("GPU available:", torch.cuda.is_available())
print("Using device :", device)
print("=" * 60)


# =====================================================
# TEXT PREPROCESSING
# =====================================================

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

# =====================================================
# CLEAN INPUT DATA
# =====================================================

def ausbildungssuche_imp_data(df_ausbildungssuche):

    df_ausbildungssuche_imp = (
        df_ausbildungssuche[
            ['angebot_titel_clean']
        ]
        .drop_duplicates()
        .copy()
    )

    return df_ausbildungssuche_imp


# =====================================================
# ENCODE BERUFENET OCCUPATIONS
# =====================================================

def encoding_berufenet(model):

    print("\nEncoding KldB occupations...")

    df_unique_bndata = (
        occupations_fur_berufsausbildung()
    )

    df_unique_bndata["Berufsbenennungen_clean"] = df_unique_bndata["Berufsbenennungen"].astype(str).apply(preprocess_text)

    berufenet_titles = (
        df_unique_bndata[
            "Berufsbenennungen_clean"
        ]
        .astype(str)
        .tolist()
    )

    berufenet_ids = (
        df_unique_bndata[
            "KldB-2010_(5-Steller)"
        ]
        .tolist()
    )

    berufenet_embeddings = model.encode(
        berufenet_titles,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return (
        berufenet_titles,
        berufenet_ids,
        berufenet_embeddings
    )


# =====================================================
# GPU SBERT MATCHING
# =====================================================

def matches_ausbildungssuche_berufenet_SBERT(
    model,
    df_ausbildungssuche
):

    results = []

    print(
        "\nEncoding Ausbildung titles "
        "and matching to KldB occupations..."
    )

    # =================================================
    # PREPARE DATA
    # =================================================

    # df_ausbildungssuche['angebot_titel_clean'] = df_ausbildungssuche["angebot.titel"].astype(str).apply(preprocess_text)

    df_ausbildungssuche_imp = (
        ausbildungssuche_imp_data(
            df_ausbildungssuche
        )
    )

    aus_titles = (
        df_ausbildungssuche_imp[
            "angebot_titel_clean"
        ]
        .astype(str)
        .tolist()
    )

    (
        berufenet_titles,
        berufenet_ids,
        berufenet_embeddings
    ) = encoding_berufenet(model)

    # =================================================
    # ENCODE AUSBILDUNG TITLES
    # =================================================

    print("\nEncoding Ausbildung titles...")

    aus_embeddings = model.encode(
        aus_titles,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # =================================================
    # MOVE EMBEDDINGS TO GPU
    # =================================================

    print("\nMoving embeddings to GPU...")

    aus_tensor = torch.tensor(
        aus_embeddings,
        device=device
    )

    berufenet_tensor = torch.tensor(
        berufenet_embeddings,
        device=device
    )

    # =================================================
    # GPU COSINE SIMILARITY MATCHING
    # =================================================

    print("\nRunning GPU similarity matching...")

    MATCHING_BATCH_SIZE = 512

    best_indices_all = []
    best_scores_all = []

    for i in tqdm(
        range(
            0,
            len(aus_tensor),
            MATCHING_BATCH_SIZE
        ),
        desc="Matching"
    ):

        batch = aus_tensor[
            i:i + MATCHING_BATCH_SIZE
        ]

        # -----------------------------------------
        # GPU cosine similarity
        # -----------------------------------------

        sims = torch.matmul(
            batch,
            berufenet_tensor.T
        )

        # -----------------------------------------
        # Top-1 nearest neighbor
        # -----------------------------------------

        best_scores, best_indices = torch.max(
            sims,
            dim=1
        )

        best_scores_all.extend(
            best_scores.cpu().numpy()
        )

        best_indices_all.extend(
            best_indices.cpu().numpy()
        )

    # =================================================
    # BUILD RESULT DATAFRAME
    # =================================================

    print("\nPreparing result dataframe...")

    for title, idx, score in zip(
        aus_titles,
        best_indices_all,
        best_scores_all
    ):

        results.append({

            "ausbildung_title":
                title,

            "best_kldb_occupation":
                berufenet_titles[idx],

            "similarity":
                float(score),

            "Kldb_code":
                berufenet_ids[idx]
        })

    df_sbert_matches = pd.DataFrame(
        results
    )

    return df_sbert_matches


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":

    print("\nLoading Ausbildungssuche dataset...")
    path_raw_data_ausbildungssuche = '/home/gshaik@forschungsnetz.local/project/data_1/raw_data/Ausbildungssuche_complete_data.csv'
    df_ausbildungssuche = pd.read_csv(path_raw_data_ausbildungssuche)
    df_ausbildungssuche.drop_duplicates(subset="id", inplace=True) # Removing duplicates based on 'id'
    df_ausbildungssuche['angebot_titel_clean'] = df_ausbildungssuche["angebot.titel"].astype(str).apply(preprocess_text)

    print("\nLoading SBERT model...")
    model = SentenceTransformer("Sahajtomar/German-semantic", device=device)
    print("Model loaded successfully")

    df_matches = matches_ausbildungssuche_berufenet_SBERT(model, df_ausbildungssuche)

    print("\nMerging results...")
    df_ausbildungssuche_kldb = pd.merge(df_ausbildungssuche, df_matches, left_on="angebot_titel_clean", right_on="ausbildung_title", how="left")
    df_ausbildungssuche_kldb.drop("angebot_titel_clean", axis=1, inplace=True)
    output_path = '/home/gshaik@forschungsnetz.local/project/data/output/ausbildungssuche_kldb_assigned.csv'
    df_ausbildungssuche_kldb.to_csv(output_path, index=False)

    print("\n Results saved successfully")
    print(f"Saved to: {output_path}")

    if os.path.exists(path_raw_data_ausbildungssuche):
        os.remove(path_raw_data_ausbildungssuche)
        print(f'{path_raw_data_ausbildungssuche} deleted')
    else:
        print(f"{path_raw_data_ausbildungssuche} does not exist.")