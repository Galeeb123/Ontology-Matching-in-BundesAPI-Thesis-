from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from kldb_input import kldb_occupations


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


def jobsuche_imp_data(df_jobsuche):

    df_jobsuche_imp = (
        df_jobsuche[
            ['beruf']
        ]
        .drop_duplicates()
        .copy()
    )

    return df_jobsuche_imp


# =====================================================
# ENCODE KldB OCCUPATIONS
# =====================================================

def encoding_berufenet(model):

    print("\nEncoding KldB occupations...")

    df_unique_bndata = (
        kldb_occupations()
    )

    berufenet_titles = (
        df_unique_bndata[
            "Berufsbenennungen"
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

def matches_jobsuche_berufenet_SBERT(
    model,
    df_jobsuche
):

    results = []

    print(
        "\nEncoding jobsuche titles "
        "and matching to KldB occupations..."
    )

    # =================================================
    # PREPARE DATA
    # =================================================

    df_jobsuche_imp = (
        jobsuche_imp_data(
            df_jobsuche
        )
    )

    job_titles = (
        df_jobsuche_imp[
            "beruf"
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
    # ENCODE JOB TITLES
    # =================================================

    print("\nEncoding jobsuche titles...")

    job_embeddings = model.encode(
        job_titles,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # =================================================
    # MOVE EMBEDDINGS TO GPU
    # =================================================

    print("\nMoving embeddings to GPU...")

    job_tensor = torch.tensor(
        job_embeddings,
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
            len(job_tensor),
            MATCHING_BATCH_SIZE
        ),
        desc="Matching"
    ):

        batch = job_tensor[
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
        job_titles,
        best_indices_all,
        best_scores_all
    ):

        results.append({

            "job_title":
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

    df_jobsuche = pd.read_csv('/home/gshaik@forschungsnetz.local/project/data/raw_data/Jobsuche_complete_data.csv')
    df_jobsuche.drop_duplicates(subset="refnr", inplace=True) # Removing duplicates based on 'refnr'
    model = SentenceTransformer("Sahajtomar/German-semantic", device=device)
    df_matches = matches_jobsuche_berufenet_SBERT(model, df_jobsuche)
    df_jobsuche_kldb = pd.merge(df_jobsuche, df_matches, left_on="beruf", right_on="job_title", how="left")
    df_jobsuche_kldb.to_csv('/home/gshaik@forschungsnetz.local/project/data/output/jobsuche_kldb_assigned.csv',index=False)
    print("\n Results saved successfully")