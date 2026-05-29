import requests
import pandas as pd 
from tqdm import tqdm
import numpy as np

# Header with API key
#header_entgeltatlas = {
#    'X-API-Key': 'sete-inspirieren'
#}

def pages_kldb_entgelt(header):
    # Target URL
    url_entgeltatlas = 'https://rest.arbeitsagentur.de/infosysbub/dkz-rest/pc/v1//kldb2010?page=1'

    try:
        # GET request with 60-second timeout
        response = requests.get(url_entgeltatlas, headers=header, timeout=60)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        data_entgeltatlas = response.json()       # Parse JSON response
        # print(data_entgelte)
        pages_entgelt = data_entgeltatlas['page']['totalPages']
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    return pages_entgelt


def kldb_codes_entgelt(header):
    all_kldb_codes = []
    print(f'Number of pages of Kldb codes: {pages_kldb_entgelt(header)}')
    print('All kldb codes for entgeltatlas are being retrived....')
    for i in tqdm(range(1, pages_kldb_entgelt(header)+1)):
        # Define the API endpoint
        url = f'https://rest.arbeitsagentur.de/infosysbub/dkz-rest/pc/v1//kldb2010?page={i}'
        
        try:
            response = requests.get(url, headers=header, timeout=200)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            data_kldb = response.json()       # Parse JSON response
            df_data_kldb = pd.DataFrame(data_kldb['_embedded']['berufssystematiken'])
        except requests.exceptions.RequestException as e:
            print('Total number of pages not retrived')
            print(f"Error: {e}")
        all_kldb_codes.append(df_data_kldb)

    all_kldb_codes_df = pd.concat(all_kldb_codes, ignore_index=True)
    all_kldb_codes_df['kldb_number'] = all_kldb_codes_df['codenr'].str.replace('B ', '')
    array_kldb_numbers = np.array(all_kldb_codes_df['kldb_number'])
    return array_kldb_numbers


def data_entgelt(header):
    all_data = []
    for kldb in tqdm(kldb_codes_entgelt(header)):
        # Target URL
        url = f'https://rest.arbeitsagentur.de/infosysbub/entgeltatlas/pc/v1/entgelte/{kldb}'

        try:
            # GET request with 60-second timeout
            response = requests.get(url, headers=header, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            data_entgelte = response.json()       # Parse JSON response
            #print(data_entgelte)
            if len(data_entgelte)>0:
                data_entgelte_df = pd.json_normalize(data_entgelte)
                data_entgelte_df_cleaned = data_entgelte_df[data_entgelte_df['entgelt']>1]
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
 
        if len(data_entgelte_df_cleaned) >0:
            all_data.append(data_entgelte_df_cleaned)
            # print(f'Total no. of records for the kldb: {kldb}: {len(data_entgelte_df_cleaned)}')
    entgelt_data = pd.concat(all_data, ignore_index=True)
    entgelt_data.to_csv('/home/gshaik@forschungsnetz.local/project/data/raw_data/entgeltatlas_complete_data.csv', index=False)
    print('entgeltatlas data retrived')
    return entgelt_data


# print(kldb_codes_entgelt(header_entgeltatlas))
