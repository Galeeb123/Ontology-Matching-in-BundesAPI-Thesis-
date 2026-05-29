import requests
import pandas as pd
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from sentence_transformers import SentenceTransformer
import torch

from sqlalchemy import create_engine
from datetime import datetime
from uuid import uuid4
import os

from berufenet_data_retrival import berufenet_data
from entgeltatlas_data_retrival import data_entgelt
from Ausbildungssuche_data_retrival import data_ausbildungssuche, count_records_ausbildungssuche
from jobsuche_data_retrival import jobsuche_data
from bewerberboerse_data_retrival import bewerberboerse_data

from data_wearhouse_ausbildungssuche import data_push_ausbildungssuche
from data_wearhouse_bewerberboerse import data_push_bewerberboerse
from data_wearhouse_entgeltatlas import data_push_entgeltatlas
from data_wearhouse_jobsuche import data_push_jobsuche

from kldb_input import kldb_occupations, occupations_fur_berufsausbildung
from kldb_ausbildungssuche import matches_ausbildungssuche_berufenet_SBERT, preprocess_text
from kldb_bewerberboerse import matches_bewerberboerse_berufenet_SBERT
from kldb_jobsuche import matches_jobsuche_berufenet_SBERT

# from kldb_berufenet import Title_occupation, assign_kldb_occupation_berufenet
# from kldb_entgeltatlas import avg_salary

# ==============================================================================
# Data Retriving functions
# ==============================================================================

# +*+*+* ----------------------------------------------------------------- +*+*+*

def run_berufenet():
    headers_bnet = {"X-API-Key": "infosysbub-berufenet"}
    berufenet_data(headers_bnet)
    
# +*+*+* ----------------------------------------------------------------- +*+*+*

def run_entgeltatlas():
    header_entgeltatlas = {'X-API-Key': 'sete-inspirieren'}
    data_entgelt(header_entgeltatlas)

# +*+*+* ----------------------------------------------------------------- +*+*+*

def run_ausbildungssuche():
    header_absuche = {"X-API-Key": "infosysbub-absuche"}
    regions = ['BAW','BAY','BER','BRA','BRE','HAM','HES','MBV','NDS','NRW','RPF','SAA','SAC','SAN','SLH','TH%C3%9C','-']
    beginntermine = [2,101,102,103,104,105,106,107,108,109,110,111,112]
    data_ausbildungssuche(header_absuche, regions, beginntermine)

# +*+*+* ----------------------------------------------------------------- +*+*+*

def run_jobsuche():
    header_jobsuche = {"X-API-Key": "jobboerse-jobsuche"}
    jobsuche_data(header_jobsuche)

# +*+*+* ----------------------------------------------------------------- +*+*+*

def run_bewerberboerse():
    header_bewerberboerse = {"X-API-Key": "jobboerse-bewerbersuche-ui"}
    bewerberboerse_data(header_bewerberboerse)

# +*+*+* ----------------------------------------------------------------- +*+*+*
'''
with ThreadPoolExecutor(max_workers=5) as executor:
    executor.submit(run_berufenet)
    executor.submit(run_entgeltatlas)
    executor.submit(run_ausbildungssuche)
    executor.submit(run_jobsuche)
    executor.submit(run_bewerberboerse)
'''

if __name__ == "__main__":

    # +*+*+* ----------------------------------------------------------------- +*+*+*
    
    print('Data retriving started')

    print('Retriving berufenet data...')
    run_berufenet()
    print('Retrived berufenet data')
    
    print('Retriving entdeltatlas data...')
    run_entgeltatlas()
    print('Retrived entdeltatlas data')
    
    print('Retriving ausbildungs data...')
    run_ausbildungssuche()
    print('Retrived ausbildungs data')

    print('Retriving jobsuche data...')
    run_jobsuche()
    print('Retrived jobsuche data')
 
    print('Retriving bewerberboerse data...')
    run_bewerberboerse()
    print('Retrived bewerberboerse data')

    print('Data retriving ended')
    
    # +*+*+* ----------------------------------------------------------------- +*+*+*
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
    )
    print("=" * 60)
    print("GPU available:", torch.cuda.is_available())
    print("Using device :", device)
    print("=" * 60)

    print('Assigning KldB codes')

            # +*+*+* ----------------------------------------------------------------- +*+*+*
    print('Assigning KldB codes for Ausbildungssuche')

    path_raw_data_ausbildungssuche = '/home/gshaik@forschungsnetz.local/project/data/raw_data/Ausbildungssuche_complete_data.csv'
    df_ausbildungssuche = pd.read_csv(path_raw_data_ausbildungssuche)
    df_ausbildungssuche.drop_duplicates(subset="id", inplace=True) # Removing duplicates based on 'id'
    df_ausbildungssuche['angebot_titel_clean'] = df_ausbildungssuche["angebot.titel"].astype(str).apply(preprocess_text)
    model = SentenceTransformer("Sahajtomar/German-semantic", device=device)
    df_matches_ausbildungssuche = matches_ausbildungssuche_berufenet_SBERT(model, df_ausbildungssuche)
    df_ausbildungssuche_kldb = pd.merge(df_ausbildungssuche, df_matches_ausbildungssuche, left_on="angebot_titel_clean", right_on="ausbildung_title", how="left")
    df_ausbildungssuche_kldb.drop("angebot_titel_clean", axis=1, inplace=True)

    output_path_ausbildungssuche = '/home/gshaik@forschungsnetz.local/project/data/output/ausbildungssuche_kldb_assigned.csv'
    df_ausbildungssuche_kldb.to_csv(output_path_ausbildungssuche, index=False)
    print(f"Results Saved to: {output_path_ausbildungssuche}")

    if os.path.exists(path_raw_data_ausbildungssuche):
        os.remove(path_raw_data_ausbildungssuche)
        print(f'{path_raw_data_ausbildungssuche} deleted')
    else:
        print(f"{path_raw_data_ausbildungssuche} does not exist.")

            # +*+*+* ----------------------------------------------------------------- +*+*+*
    
    print('Assigning KldB codes for Jobsuche')

    path_raw_data_jobsuche = '/home/gshaik@forschungsnetz.local/project/data/raw_data/Jobsuche_complete_data.csv'
    df_jobsuche = pd.read_csv(path_raw_data_jobsuche)
    df_jobsuche.drop_duplicates(subset="refnr", inplace=True) # Removing duplicates based on 'refnr'
    df_jobsuche['beruf_clean'] = df_jobsuche["beruf"].astype(str).apply(preprocess_text)
    model = SentenceTransformer("Sahajtomar/German-semantic", device=device)
    df_matches_jobsuche = matches_jobsuche_berufenet_SBERT(model, df_jobsuche)
    df_jobsuche_kldb = pd.merge(df_jobsuche, df_matches_jobsuche, left_on="beruf_clean", right_on="job_title", how="left")
    df_jobsuche_kldb.drop("beruf_clean", axis=1, inplace=True)
    df_jobsuche_kldb.to_csv('/home/gshaik@forschungsnetz.local/project/data/output/jobsuche_kldb_assigned.csv',index=False)

    if os.path.exists(path_raw_data_jobsuche):
        os.remove(path_raw_data_jobsuche)
        print(f'{path_raw_data_jobsuche} deleted')
    else:
        print(f"{path_raw_data_jobsuche} does not exist.")

            # +*+*+* ----------------------------------------------------------------- +*+*+*
    
    print('Assigning KldB codes for bewerberboerse')

    path_raw_data_bewerberboerse = '/home/gshaik@forschungsnetz.local/project/data/raw_data/bewerberboerse_complete_data.csv'
    df_bewerberboerse = pd.read_csv(path_raw_data_bewerberboerse)
    df_bewerberboerse.drop_duplicates(subset="refnr", inplace=True) # Removing duplicates based on 'refnr'
    df_bewerberboerse['berufe_clean'] = df_bewerberboerse["berufe"].astype(str).apply(preprocess_text)
    model = SentenceTransformer("Sahajtomar/German-semantic", device=device)
    df_matches_bewerberboerse = matches_bewerberboerse_berufenet_SBERT(model, df_bewerberboerse)
    df_bewerberboerse_kldb = pd.merge(df_bewerberboerse, df_matches_bewerberboerse, left_on="berufe_clean",right_on="job_title",how="left")
    df_bewerberboerse_kldb.drop("berufe_clean", axis=1, inplace=True)
    df_bewerberboerse_kldb.to_csv('/home/gshaik@forschungsnetz.local/project/data/output/bewerberboerse_kldb_assigned.csv', index=False)

    if os.path.exists(path_raw_data_bewerberboerse):
        os.remove(path_raw_data_bewerberboerse)
        print(f'{path_raw_data_bewerberboerse} deleted')
    else:
        print(f"{path_raw_data_bewerberboerse} does not exist.")

    Print('Assigned KldB codes')
    
    # +*+*+* ----------------------------------------------------------------- +*+*+*
    print('Data pushing started')
    # credentials for PostgreSQL
    DB_USER = "ext_gshaik"
    DB_PASS = "Galeeb123"
    DB_HOST = "10.15.0.12"
    DB_PORT = "5432"
    DB_NAME = "dwh_radar_dev"

    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    print('engine launched to push bewerberboerse data')

    file_path_ausbildungssuche = "/home/gshaik@forschungsnetz.local/project/data/output/ausbildungssuche_kldb_assigned.csv"
    data_push_ausbildungssuche(file_path_ausbildungssuche)
    
    file_path_bewerberboerse = "/home/gshaik@forschungsnetz.local/project/data/output/bewerberboerse_kldb_assigned.csv"
    data_push_bewerberboerse(file_path_bewerberboerse)
    
    file_path_entgeltatlas = "/home/gshaik@forschungsnetz.local/project/data/raw_data/entgeltatlas_complete_data.csv"
    data_push_entgeltatlas(file_path_entgeltatlas)

    file_path_jobsuche = "/home/gshaik@forschungsnetz.local/project/data/output/jobsuche_kldb_assigned.csv"
    data_push_jobsuche(file_path_jobsuche)

    print('Data pushing ended')



