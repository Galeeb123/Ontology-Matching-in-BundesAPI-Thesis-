
 
# =====================================================
# IMPORTS
# =====================================================

import re
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from kldb_input import kldb_occupations


# =====================================================
# GPU CONFIGURATION
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("GPU available:", torch.cuda.is_available())
print("Using device :", device)
print("=" * 70)


# =====================================================
# HYPERPARAMETERS
# =====================================================

MODEL_NAME = "Sahajtomar/German-semantic"

EMBEDDING_BATCH_SIZE = 64
MATCHING_BATCH_SIZE = 512

MAX_SEQ_LENGTH = 128


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
    # Normalize occupational suffixes
    # entwickler/in -> entwickler
    # -----------------------------------------

    # text = re.sub(r'/in\b', '', text)

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
# LOAD DATA
# =====================================================

print("\nLoading datasets...")

# Validation dataset
df_validation = pd.read_csv(
    "/home/galeebs/project/input/ojas.csv"
)

# Reference ontology dataset
df_input = kldb_occupations()

print("Validation rows:", len(df_validation))
print("Reference rows :", len(df_input))


# =====================================================
# CLEAN KldB CODES
# =====================================================

print("\nCleaning KldB codes...")

df_validation["f100_kldb2010"] = (
    df_validation["f100_kldb2010"]
    .astype(str)
    .str.extract(r'(\d{5})')[0]
)

df_input["KldB-2010_(5-Steller)"] = (
    df_input["KldB-2010_(5-Steller)"]
    .astype(str)
    .str.extract(r'(\d{5})')[0]
)


# =====================================================
# REMOVE MISSING VALUES
# =====================================================

print("\nRemoving missing values...")

df_validation = df_validation.dropna(
    subset=["TF100", "f100_kldb2010"]
)

df_input = df_input.dropna(
    subset=["Berufsbenennungen", "KldB-2010_(5-Steller)"]
)

print("Validation rows after cleaning:",
      len(df_validation))

print("Reference rows after cleaning :",
      len(df_input))


# =====================================================
# PREPROCESS TEXT
# =====================================================

print("\nPreprocessing German occupational titles...")

df_validation["TF100_clean"] = (
    df_validation["TF100"]
    .astype(str)
    .apply(preprocess_text)
)

df_input["Berufsbenennungen_clean"] = (
    df_input["Berufsbenennungen"]
    .astype(str)
    .apply(preprocess_text)
)

print("\nSample preprocessing examples:")

for i in range(min(5, len(df_validation))):

    original = df_validation.iloc[i]["TF100"]
    cleaned = df_validation.iloc[i]["TF100_clean"]

    print(f"\nOriginal : {original}")
    print(f"Processed: {cleaned}")


# =====================================================
# LOAD GERMAN SBERT MODEL
# =====================================================

print("\nLoading German SBERT model...")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)

model.max_seq_length = MAX_SEQ_LENGTH

print("Model loaded successfully")
print("Model:", MODEL_NAME)


# =====================================================
# PREPARE TITLES
# =====================================================

validation_titles = (
    df_validation["TF100_clean"]
    .astype(str)
    .tolist()
)

input_titles = (
    df_input["Berufsbenennungen_clean"]
    .astype(str)
    .tolist()
)


# =====================================================
# GENERATE EMBEDDINGS
# =====================================================

print("\nGenerating validation embeddings...")

validation_embeddings = model.encode(
    validation_titles,
    batch_size=EMBEDDING_BATCH_SIZE,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("\nGenerating reference embeddings...")

input_embeddings = model.encode(
    input_titles,
    batch_size=EMBEDDING_BATCH_SIZE,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("\nEmbedding generation completed")

print("Validation embedding shape:",
      validation_embeddings.shape)

print("Reference embedding shape:",
      input_embeddings.shape)


# =====================================================
# MOVE EMBEDDINGS TO GPU
# =====================================================

print("\nMoving embeddings to GPU...")

validation_tensor = torch.tensor(
    validation_embeddings,
    device=device
)

input_tensor = torch.tensor(
    input_embeddings,
    device=device
)


# =====================================================
# NORMALIZE EMBEDDINGS
# =====================================================

print("\nNormalizing embeddings...")

validation_tensor = torch.nn.functional.normalize(
    validation_tensor,
    p=2,
    dim=1
)

input_tensor = torch.nn.functional.normalize(
    input_tensor,
    p=2,
    dim=1
)


# =====================================================
# GPU COSINE SIMILARITY MATCHING
# =====================================================

print("\nRunning GPU similarity matching...")

predicted_kldb = []
similarity_scores = []

for i in tqdm(
    range(0, len(validation_tensor), MATCHING_BATCH_SIZE),
    desc="Matching"
):

    batch = validation_tensor[i:i + MATCHING_BATCH_SIZE]

    # -----------------------------------------
    # Cosine similarity matrix
    # -----------------------------------------

    sims = torch.matmul(
        batch,
        input_tensor.T
    )

    # -----------------------------------------
    # Best similarity scores
    # -----------------------------------------

    best_scores, best_indices = torch.max(
        sims,
        dim=1
    )

    # -----------------------------------------
    # Move to CPU
    # -----------------------------------------

    best_scores = best_scores.cpu().numpy()
    best_indices = best_indices.cpu().numpy()

    # -----------------------------------------
    # Save predictions
    # -----------------------------------------

    for idx, score in zip(best_indices, best_scores):

        predicted_kldb.append(
            df_input.iloc[idx]["KldB-2010_(5-Steller)"]
        )

        similarity_scores.append(float(score))


# =====================================================
# SAVE PREDICTIONS
# =====================================================

df_validation["predicted_kldb"] = predicted_kldb

df_validation["similarity_score"] = similarity_scores


# =====================================================
# EVALUATION
# =====================================================

print("\nEvaluating model performance...")

y_true = (
    df_validation["f100_kldb2010"]
    .astype(str)
)

y_pred = (
    df_validation["predicted_kldb"]
    .astype(str)
)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# =====================================================
# PRINT RESULTS
# =====================================================

print("\n" + "=" * 70)
print("📊 GERMAN SBERT EVALUATION RESULTS")
print("=" * 70)

print(f"Model                : {MODEL_NAME}")
print(f"Embedding Batch Size : {EMBEDDING_BATCH_SIZE}")
print(f"Matching Batch Size  : {MATCHING_BATCH_SIZE}")
print(f"Max Sequence Length  : {MAX_SEQ_LENGTH}")

print("-" * 70)

print(f"Accuracy             : {accuracy:.4f}")
print(f"Precision            : {precision:.4f}")
print(f"Recall               : {recall:.4f}")
print(f"F1 Score             : {f1:.4f}")

print("-" * 70)

print(f"Total Validation Rows: {len(df_validation)}")

print("=" * 70)


# =====================================================
# SAVE RESULTS
# =====================================================

output_path = (
    "/home/galeebs/project/data/performance_ojas_5_digit/"
    "performance_SBERT_improved_2.csv"
)

df_validation.to_csv(
    output_path,
    index=False
)

print("\n Results saved successfully")
print(f"Saved to: {output_path}")