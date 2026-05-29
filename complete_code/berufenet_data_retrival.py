import requests
import pandas as pd
from tqdm import tqdm
import numpy as np


def bnet_data(headers_bnet):
    # Define the API endpoint
    url_bnet = "https://rest.arbeitsagentur.de/infosysbub/bnet/pc/v1/berufe"

    # Make the GET request with a 60-second timeout
    try:
        response = requests.get(url_bnet, headers=headers_bnet, timeout=60)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        data_berufe_id = response.json()       # Parse JSON response
        # print(data_berufe_id)
        pages = data_berufe_id['page']['totalPages']
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
    return pages


def berufeliste_berufenet(headers_bnet):
    # Define the API endpoint
    total_pages = bnet_data(headers_bnet)
    # total_pages = 5
    print(f'Number of pages in Berufenet: {total_pages}')
    print('Retriving Berufenet IDs....')

    all_data=[]
    for i in tqdm(range(total_pages)):
        
        url_bnet_list = f"https://rest.arbeitsagentur.de/infosysbub/bnet/pc/v1/berufe?suchwoerter=*&page={i}"
        # Make the GET request with a 60-second timeout
        try:
            response = requests.get(url_bnet_list, headers=headers_bnet, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            data_berufe_id = response.json()       # Parse JSON response
            temp = data_berufe_id['_embedded']['berufSucheList']
            
            berufe = pd.DataFrame(temp)
            all_data.append(berufe)
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")

    final_df = pd.concat(all_data, ignore_index=True)
    ids = np.array(final_df['id'])
    return ids


def berufenet_data(headers_bnet):
    # Define the API endpoint
    count = 0
    ids_berufenet = berufeliste_berufenet(headers_bnet)
    print(f'Number of Berufenet IDs: {len(ids_berufenet)}')
    print('Retriving Berufenet Data ....')
    complete_data=[]
    for j in tqdm(ids_berufenet):
        count = count+1
        # print(count)
        # first_id = ids[j]
        # print(i)
        url_berufenet_data = f"https://rest.arbeitsagentur.de/infosysbub/bnet/pc/v1/berufe/{j}"
        # print(url)
        # Make the GET request with a 60-second timeout
        try:
            response = requests.get(url_berufenet_data, headers=headers_bnet, timeout=60)
            response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
            data = response.json()       # Parse JSON response

            df_1 = data[0]
            df_temp = pd.json_normalize(df_1)

            # berufe = pd.DataFrame(df_temp)
            complete_data.append(df_temp)
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
        # if count%100==0:
            # print(f"{count} records retrived")
    final_df_complete = pd.concat(complete_data, ignore_index=True)
    final_df_complete.to_csv('/home/gshaik@forschungsnetz.local/project/data/raw_data/berufenet_complete_data.csv', index=False)
    print('Berufenet Data retrived')
    return final_df_complete

if __name__ == "__main__":
    # Set headers including your API key
    headers_bnet = {
       "X-API-Key": "infosysbub-berufenet"
        }
    berufenet_data(headers_bnet)
#data_bnet.to_csv('/home/gshaik@forschungsnetz.local/project/data/raw_data/berufenet_complete_data.csv', index=False)
# print(data_bnet)