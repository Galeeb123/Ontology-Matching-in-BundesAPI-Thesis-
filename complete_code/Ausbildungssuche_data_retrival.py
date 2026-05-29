import requests
import pandas as pd 
from tqdm import tqdm
import numpy as np

'''
Regions and beginntermins are used to extract the complete data from Ausbildungssuche
Otherwise only first 10000 records will be retrived
'''
# Set headers including your API key
# header = {
#    "X-API-Key": "infosysbub-absuche"
#    }



def count_records_ausbildungssuche(header, regions, beginntermins):
    total_records = 0
    for region in tqdm(regions):
        total_records_region = 0
        for beginntermin in beginntermins:
            # Define the API endpoint
            url = f"https://rest.arbeitsagentur.de/infosysbub/absuche/pc/v1/ausbildungsangebot?re={region}&bt={beginntermin}"
            # url = "https://rest.arbeitsagentur.de/infosysbub/absuche/pc/v1/ausbildungsangebot"

            # Make the GET request with a 60-second timeout
            try:
                response = requests.get(url, headers=header, timeout=60)
                response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                data_Ausbildungssuche_bt = response.json()       # Parse JSON response
                records = data_Ausbildungssuche_bt['page']['totalElements']
                
                total_records_region = total_records_region + records
            except requests.exceptions.RequestException as e:
                print(f"Error: {e}")
        total_records = total_records + total_records_region
        # print(f'Total records in ausbildungssuche from region: {region}: {total_records_region}')
    print(f'Total records in ausbildungssuche: {total_records}')
    return total_records




def data_ausbildungssuche(header, regions, beginntermins):

    # total_records = 0
    print('Ausbildungssuche data is being retrived ......')
    all_data_regions = []
    for region in tqdm(regions):
        # total_records_region = 0
        all_data_beginntermin = []
        for beginntermin in beginntermins:
            # Define the API endpoint
            url_1 = f"https://rest.arbeitsagentur.de/infosysbub/absuche/pc/v1/ausbildungsangebot?re={region}&bt={beginntermin}"
            
            # Make the GET request with a 60-second timeout
            try:
                response_1 = requests.get(url_1, headers=header, timeout=200)
                response_1.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                data = response_1.json()       # Parse JSON response
                total_pages = data['page']['totalPages']
                # print(f'total_pages: {total_pages}')
            except requests.exceptions.RequestException as e:
                # print('Total number of pages not retrived')
                print(f"Error: {e}")

            if total_pages !=0:
                all_data_pages = []
                for page in range(total_pages):
                    url_2 = f"https://rest.arbeitsagentur.de/infosysbub/absuche/pc/v1/ausbildungsangebot?re={region}&bt={beginntermin}&page={page}"
                    # print(url_2)
                    try:
                        response_2 = requests.get(url_2, headers=header, timeout=60)
                        response_2.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                        data_Ausbildungssuche = response_2.json()       # Parse JSON response
                        df_ausbildungssuche_normalized_page = pd.json_normalize(data_Ausbildungssuche['_embedded']['termine'])

                        # df_ausbildungssuche_normalized_page
                        all_data_pages.append(df_ausbildungssuche_normalized_page)
                        # print(data_Ausbildungssuche)
                    except requests.exceptions.RequestException as e:
                        print(f"Error: {e}")
                if len(all_data_pages) != 0:
                    all_data_pages_df = pd.concat(all_data_pages, ignore_index=True)
                    all_data_beginntermin.append(all_data_pages_df)
        if len(all_data_beginntermin) != 0:
            all_data_beginntermin_df = pd.concat(all_data_beginntermin, ignore_index=True)
            all_data_regions.append(all_data_beginntermin_df)

        # total_records = total_records + total_records_region
        # print(f'Total records in ausbildungssuche from region: {region}: {total_records_region}')
    # print(f'Total records in ausbildungssuche: {total_records}')
    if len(all_data_regions) != 0:
        complete_data_df = pd.concat(all_data_regions, ignore_index=True)
    complete_data_df.to_csv('/home/gshaik@forschungsnetz.local/project/data/raw_data/Ausbildungssuche_complete_data.csv')
    print('Ausbildungssuche data retrived')
    return complete_data_df


# print(f'Total records in ausbildungssuche: {count_records_ausbildungssuche()}')