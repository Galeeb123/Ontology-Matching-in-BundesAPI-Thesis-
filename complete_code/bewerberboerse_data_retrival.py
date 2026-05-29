import requests
import pandas as pd 
from tqdm import tqdm
import numpy as np
 
'''
Multiple filters has been used to extract the complete data from bewerberboerse
Otherwise only first 10000 records will be retrived
'''

header = {
    "X-API-Key": "jobboerse-bewerbersuche-ui"
    }

def metadata_bewerberboerse(header):
    url = "https://rest.arbeitsagentur.de/jobboerse/bewerbersuche-service/pc/v1/bewerber"
    
    try:
        response = requests.get(url, headers=header, timeout=60)
        response.raise_for_status()
        Bewerberboerse = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    Bewerberboerse_locations = list(Bewerberboerse['facetten']['arbeitsorte']['counts'].keys())
    Bewerberboerse_berufsfelds = list(Bewerberboerse['facetten']['berufsfeld']['counts'].keys())
    Bewerberboerse_berufserfahrung = list(Bewerberboerse['facetten']['berufserfahrung']['counts'].keys())
    Bewerberboerse_sprache = list(Bewerberboerse['facetten']['sprachen_grund']['counts'].keys())

    print(f'Total number of records in bewerberboerse available: {Bewerberboerse["maxErgebnisse"]}')
    return Bewerberboerse['maxErgebnisse'], Bewerberboerse_locations, Bewerberboerse_berufsfelds, Bewerberboerse_berufserfahrung, Bewerberboerse_sprache



def bewerberboerse_data(header):
    total_records_bewerberboerse, Bewerberboerse_locations, Bewerberboerse_berufsfelds, Bewerberboerse_berufserfahrung, Bewerberboerse_sprache = metadata_bewerberboerse(header)
    try:
        print('Retriving data from bewerberboerse .....')
        all_data = []
        for job_role in tqdm(Bewerberboerse_berufsfelds):
            all_data_job_location = []
            for location in Bewerberboerse_locations:
                all_data_erfahrung = []
                for erfahrung in Bewerberboerse_berufserfahrung:
                    #all_data_sprache = []
                    #for sprache in Bewerberboerse_sprache:
                    page_count =1
                    all_data_pages = []
                    while True:
                        url = f"https://rest.arbeitsagentur.de/jobboerse/bewerbersuche-service/pc/v1/bewerber?berufsfeld={job_role}&wo={location}&berufserfahrung={erfahrung}&page={page_count}"
                        
                        try:
                            response = requests.get(url, headers=header, timeout=60)
                            response.raise_for_status()
                            jobs_field_sprache = response.json()
                            if 'bewerber' in list(jobs_field_sprache.keys()):
                                # print('there are some records')
                                if len(jobs_field_sprache['bewerber']) !=0:
                                    # print('There are definetly some records')
                                    df_jobs_field_sprache_page = pd.json_normalize(jobs_field_sprache['bewerber'])
                                    all_data_pages.append(df_jobs_field_sprache_page)
                                else:
                                    break

                            else:
                                # print('There are no records')
                                break
                        

                        except requests.exceptions.RequestException as e:
                            print(f"Request failed: {e}")

                        page_count = page_count +1
                        if page_count>400:
                                break
                    
                    if len(all_data_pages) != 0:
                        all_data_pages_df = pd.concat(all_data_pages, ignore_index=True)
                        # print(f'job_role: {job_role}, job_location: {location}, berufserfahrung: {erfahrung}, no. of records: {len(all_data_pages_df)}')
                        all_data_erfahrung.append(all_data_pages_df)
                    #if len(all_data_sprache) != 0:
                    #    all_data_sprache_df = pd.concat(all_data_sprache, ignore_index=True)
                        # print(f'job_role: {job_role}, no. of records: {len(all_data_sprache_df)}')
                    #    all_data_erfahrung.append(all_data_sprache_df)
                if len(all_data_erfahrung) != 0:
                        all_data_erfahrung_df = pd.concat(all_data_erfahrung, ignore_index=True)
                        # print(f'job_role: {job_role}, job_location: {location}, no. of records: {len(all_data_erfahrung_df)}')
                        all_data_job_location.append(all_data_erfahrung_df)
            if len(all_data_job_location) != 0:
                        all_data_job_location_df = pd.concat(all_data_job_location, ignore_index=True)
                        # print(f'job_role: {job_role}, no. of records: {len(all_data_job_location_df)}')
                        all_data.append(all_data_job_location_df)
        all_data_df = pd.concat(all_data, ignore_index=True)
        # print(f'no.of uncleaned data retived (may contain duplicates): {len(all_data_df)}')

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Exiting gracefully.")
        # df_bewerberboerse_cleaned = all_data_df.drop_duplicates()
        all_data_df.to_csv('/home/gshaik@forschungsnetz.local/project/data/raw_data/bewerberboerse_complete_data.csv')

    all_data_df.to_csv('/home/gshaik@forschungsnetz.local/project/data/raw_data/bewerberboerse_complete_data.csv')
    print('bewerberboerse data retrived')
    return all_data_df
 
# bewerberboerse_data(header)