'''
This file will push the KldB_assigned entgeltatlas data to PostgreSQL
Then will delete the existing file ../data/output/entgeltatlas_complete_data
'''

import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from uuid import uuid4
import os

file_path_entgeltatlas = "/home/gshaik@forschungsnetz.local/project/data/raw_data/entgeltatlas_complete_data.csv"

# credentials for PostgreSQL
DB_USER = "ext_gshaik"
DB_PASS = "Galeeb123"
DB_HOST = "10.15.0.12"
DB_PORT = "5432"
DB_NAME = "dwh_radar_dev"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
 
print('engine launched to push entgeltatlas data')

def data_push_entgeltatlas(path):
    df_entgeltatlas = pd.read_csv(path)

    # Unique ID for this run
    run_id = str(uuid4())

    # Timestamp for this run
    created_at = datetime.now()

    df_entgeltatlas["run_id"] = run_id
    df_entgeltatlas["created_at"] = created_at



    print('Pushing entgeltatlas data to PostgreSQL')

    df_entgeltatlas.to_sql(
        name="entgeltatlas_complete_data",
        con=engine,
        if_exists="append",
        index=False
    )

    print("entgeltatlas data inserted successfully!")

    if os.path.exists(path):
        os.remove(path)
        print(f'{path} deleted')
    else:
        print(f"{path} does not exist.")

if __name__ == "__main__":
    data_push_entgeltatlas(file_path_entgeltatlas)