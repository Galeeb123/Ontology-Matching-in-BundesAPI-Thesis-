from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from kldb_berufenet import berufenet_imp_data



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

def jobsuche_imp_data(df_jobsuche):
    df_jobsuche_imp = df_jobsuche[['beruf']]
    df_jobsuche_imp.drop_duplicates(inplace=True)
    return df_jobsuche_imp



def encoding_berufenet(model):
    # Multilingual model (works well for German occupation titles)
    df_unique_bndata = berufenet_imp_data()
    berufenet_titles = df_unique_bndata["Short_occupation_title"].tolist()
    berufenet_ids = df_unique_bndata["Kldb_numbers"].tolist()
    berufenet_embeddings = model.encode(
        berufenet_titles,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True  # cosine-friendly
        )
    return berufenet_titles, berufenet_ids, berufenet_embeddings

def matches_jobsuche_berufenet_SBERT(model, df_jobsuche):
    results = []
    print("Encoding jobsuche titles and matching to BERUFENET...")
    df_jobsuche_imp = jobsuche_imp_data(df_jobsuche)
    berufenet_titles, berufenet_ids, berufenet_embeddings = encoding_berufenet(model)
    for idx, row in tqdm(df_jobsuche_imp.iterrows(), total=len(df_jobsuche_imp)):
        job_title = row["beruf"]
        # Encode this Ausbildung title 
        job_emb = model.encode(
            job_title,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).reshape(1, -1)

        # Cosine similarity with ALL BERUFENET embeddings
        sims = cosine_similarity(job_emb, berufenet_embeddings)[0]

        # Get best match
        best_idx = sims.argmax()
        best_score = sims[best_idx]

        results.append({
            "job_title": job_title,
            "best_berufenet_title": berufenet_titles[best_idx],
            "similarity": float(best_score),
            "Kldb_berufenet": berufenet_ids[best_idx]
        })

    df_sbert_matches = pd.DataFrame(results)
    return df_sbert_matches


if __name__ == "__main__":
    df_jobsuche = pd.read_csv('/home/gshaik@forschungsnetz.local/project/data/raw_data/Jobsuche_complete_data.csv')
    df_jobsuche.drop_duplicates(subset="refnr", inplace=True)
    model = SentenceTransformer("Sahajtomar/German-semantic", device=device)
    df_matches = matches_jobsuche_berufenet_SBERT(model, df_jobsuche)
    df_jobsuche_kldb = df_merged = pd.merge(df_jobsuche, df_matches, left_on="beruf",right_on="job_title",how="left")
    df_jobsuche_kldb.to_csv('/home/gshaik@forschungsnetz.local/project/data/output/jobsuche_kldb_assigned.csv', index=False)