
# Introduction

This repository performs the following operations:
  1) This repository pulls the data from Bundesagentur für Arbeit APIs such as `JOBSUCHE`, `AUSBILDUNGSSUCHE`, `ENTGELTATLAS`, `BEWERBERBOERSE`, `BERUFENET`
  2) Assigns KldB codes using Ontology matching (Pretrained `SBERT` has been used)
  3) Saves the datasets to `PostgreSQL`

# Folder Structure

```text
project/
│
├── complete_code/
├── data/
│   ├── raw_data/
│   └── output/
├── input/
└── requirements.txt
```
# Description of files

All python libraries required for this thesis are in `requirements.txt`

## Files in complete_code folder

`main_file.py` --> This is the main file that executes the complete code


`kldb_input.py` --> This file read the `DKZ_Suchworte_Systematik_und_Berufe_gueltig.xlsx` and is used for Ontology matching

Files used to retrieve the data from APIs:
1) `berufenet_data_retrival.py` --> retrieves the data from `BERUFENET`
2) `entgeltatlas_data_retrival.py` --> retrieves the data from `ENTGELTATLAS`
3) `Ausbildungssuche_data_retrival.py` --> retrieves the data from `AUSBILDUNGSSUCHE`
4) `jobsuche_data_retrival` --> retrieves the data from `JOBSUCHE` 
5) `bewerberboerse_data_retrival` --> retrieves the data from `BEWERBERBOERSE`

Files used to assign KldB codes (Ontology matching):
1) `kldb_ausbildungssuche.py` --> assigns KldB codes to `AUSBILDUNGSSUCHE` data
2) `kldb_jobsuche.py` --> assigns KldB codes to `JOBSUCHE` data
3) `kldb_bewerberboerse.py` --> assigns KldB codes to `BEWERBERBOERSE` data

Files used to push the data to PostgreSQL:
1) `data_wearhouse_ausbildungssuche.py` --> Loads data to the table `ausbildungssuche_kldb_assigned` in PostgreSQL
2) `data_wearhouse_bewerberboerse.py` --> Loads data to the table `bewerberboerse_kldb_assigned` in PostgreSQL
3) `data_wearhouse_entgeltatlas.py` --> Loads data to the table `entgeltatlas_complete_data` in PostgreSQL
4) `data_wearhouse_jobsuche.py` --> Loads data to the table `jobsuche_kldb_assigned` in PostgreSQL

Notebooks used for data analysis (just for reference):
1) `analysis_ausbildungssuche.ipynb` --> created to analyse the `AUSBILDUNGSSUCHE` data
2) `analysis_bewerberboerse.ipynb` --> created to analyse the `BEWERBERBOERSE` data
3) `analysis_entgeltatlas.ipynb` --> created to analyse the `ENTGELTATLAS` data
4) `analysis_jobsuche.ipynb` --> created to analyse the `JOBSUCHE` data

File used to evaluate the performance of SBERT
1) `performance_SBERT.py` --> evaluates the performance of SBERT on `input/ojas.csv`

## files in data folder

This folder saves the `.csv` files temporarily before saving the data to `postgreSQL`

1) `raw_data` --> location for saving raw_data retrieved from APIs (files in it will be automatically deleted after saving in to `output`)
2) `output` --> location for saving KldB assigned data (files in it will be automatically deleted after saving in to `PostgreSQL`)

## files in input folder

1) `DKZ_Suchworte_Systematik_und_Berufe_gueltig.xlsx` --> Data used for ontology matching (has occupation titles with KldB codes)
2) `ojas.csv` --> used to validate SBERT model (online job advertisements)
