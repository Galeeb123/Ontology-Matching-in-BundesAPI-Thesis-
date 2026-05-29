import requests
import pandas as pd 
from tqdm import tqdm
import numpy as np
 
'''
Multiple filters has been used to extract the complete data from jobsuche
Otherwise only first 10000 records will be retrived
'''



def metadata_jobsuche(header):
    url = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs"
    
    try:
        response = requests.get(url, headers=header, timeout=60)
        response.raise_for_status()
        jobs = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    job_locations = list(jobs['facetten']['arbeitsort']['counts'].keys())
    job_roles = list(jobs['facetten']['berufsfeld']['counts'].keys())

    print(f'Total number of jobs available: {jobs["maxErgebnisse"]}')
    return jobs['maxErgebnisse'], job_locations, job_roles




def jobsuche_data(header):
    total_jobs, job_locations, job_roles = metadata_jobsuche(header)
    all_data = []
    print('Retriving data from Jobsuche .....')
    for job_role in tqdm(job_roles):
        all_data_job_role = []
        for location in job_locations:
            page_count =1
            all_data_pages = []
            while True:
                url = f"https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs?berufsfeld={job_role}&wo={location}&page={page_count}"
                try:
                    
                    response = requests.get(url, headers=header, timeout=60)
                    response.raise_for_status()
                    jobs_field_location = response.json()
                    # if page_count == 1:
                    #     print(f"No of job for the role {job_role} in {location}: {jobs_field_location['maxErgebnisse']}")
                    # print(jobs_field_location)
                    if 'stellenangebote' in list(jobs_field_location.keys()):
                        # print('there are some records')
                        if len(jobs_field_location['stellenangebote']) !=0:
                            # print('There are definetly some records')
                            df_jobs_field_location_page = pd.json_normalize(jobs_field_location['stellenangebote'])
                            all_data_pages.append(df_jobs_field_location_page)
                        else:
                            break

                    else:
                        # print('There are no records')
                        break
                
                except requests.exceptions.RequestException as e:
                    print(f"Request failed: {e}")

                page_count = page_count +1
            
            if len(all_data_pages) != 0:
                all_data_pages_df = pd.concat(all_data_pages, ignore_index=True)
                # print(f'job_location: {location}, job_role: {job_role}, no. of records: {len(all_data_pages_df)}')
                all_data_job_role.append(all_data_pages_df)
        if len(all_data_job_role) != 0:
            all_data_job_role_df = pd.concat(all_data_job_role, ignore_index=True)
            # print(f'job_role: {job_role}, no. of records: {len(all_data_job_role_df)}')
            all_data.append(all_data_job_role_df)
    all_data_df = pd.concat(all_data, ignore_index=True)
    # print(f'no.of uncleaned data retived (may contain duplicates): {len(all_data_df)}')


    df_jobsuche_cleaned = all_data_df.drop_duplicates()
    df_jobsuche_cleaned.to_csv('/home/gshaik@forschungsnetz.local/project/data/raw_data/Jobsuche_complete_data.csv')
    print('Jobsuche data retrived')
    return df_jobsuche_cleaned


