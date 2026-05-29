'''
This file will push the KldB_assigned bewerberboerse data to PostgreSQL
Then will delete the existing file ../data/output/bewerberboerse_kldb_assigned
'''

import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from uuid import uuid4
import os

file_path_bewerberboerse = "/home/gshaik@forschungsnetz.local/project/data/output/bewerberboerse_kldb_assigned.csv"

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

def data_push_bewerberboerse(path):
    df_bewerberboerse = pd.read_csv(path)
    df_bewerberboerse.drop_duplicates(subset="refnr", inplace=True)

    # Unique ID for this run
    run_id = str(uuid4())

    # Timestamp for this run
    created_at = datetime.now()

    df_bewerberboerse["run_id"] = run_id
    df_bewerberboerse["created_at"] = created_at

    # Renaming the column names to match with the column names in PostgreSQL
    df_bewerberboerse = df_bewerberboerse.rename(columns={
        "best_kldb_occupation": "best_berufenet_title",
        "Kldb_code": "Kldb_berufenet"
    })

    print('Pushing bewerberboerse data to PostgreSQL')

    df_bewerberboerse.to_sql(
        name="bewerberboerse_kldb_assigned",
        con=engine,
        if_exists="append",
        index=False
    )

    print("Bewerberboerse data inserted successfully!")

    if os.path.exists(path):
        os.remove(path)
        print(f'{path} deleted')
    else:
        print(f"{path} does not exist.")

if __name__ == "__main__":
    data_push_bewerberboerse(file_path_bewerberboerse)