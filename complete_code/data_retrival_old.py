import requests
import pandas as pd
from tqdm import tqdm
import numpy as np
from berufenet_data_retrival import berufenet_data
from entgeltatlas_data_retrival import data_entgelt
from Ausbildungssuche_data_retrival import data_ausbildungssuche, count_records_ausbildungssuche
from jobsuche import jobsuche_data
from bewerberboerse import bewerberboerse_data

# +*+*+* ------------------------------ +*+*+*

# Retriving berufenet data

headers_bnet = {
        "X-API-Key": "infosysbub-berufenet"
    }

data_bnet = berufenet_data(headers_bnet)
data_bnet.to_csv('../data/raw_data/berufenet_complete_data.csv', index=False)

# +*+*+* ------------------------------ +*+*+*

# Retriving entgeltatlas data

header_entgeltatlas = {
    'X-API-Key': 'sete-inspirieren'
}
data_entgeltatlas = data_entgelt(header_entgeltatlas)
data_entgeltatlas.to_csv('../data/raw_data/entgeltatlas_complete_data.csv', index=False)

# +*+*+* ------------------------------ +*+*+*

# Retriving Ausbildungssuche data

header_absuche = {
    "X-API-Key": "infosysbub-absuche"
    }
regions_ausbildungssuche = ['BAW', 'BAY', 'BER', 'BRA', 'BRE', 'HAM', 'HES', 'MBV', 'NDS', 'NRW', 'RPF', 'SAA', 'SAC', 'SAN', 'SLH', 'TH%C3%9C', '-'] # re
beginntermins_ausbildungssuche = [2, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]

# count_records_ausbildungssuche(header_absuche, regions_ausbildungssuche, beginntermins_ausbildungssuche)
data_ausbildungssuche(header_absuche, regions_ausbildungssuche, beginntermins_ausbildungssuche)

# +*+*+* ------------------------------ +*+*+*

# Retriving jobsuche data

header_jobsuche = {
        "X-API-Key": "jobboerse-jobsuche"
    }
jobsuche_data(header_jobsuche)

# +*+*+* ------------------------------ +*+*+*

# Retriving bewerberboerse data

header_bewerberboerse = {
        "X-API-Key": "jobboerse-bewerbersuche-ui"
    }
bewerberboerse_data(header_bewerberboerse)

# +*+*+* ------------------------------ +*+*+* 