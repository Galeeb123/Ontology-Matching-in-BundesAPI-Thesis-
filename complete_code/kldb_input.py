import pandas as pd
import ast

def kldb_occupations():
    data_kldb = pd.read_excel('/home/gshaik@forschungsnetz.local/project/input/DKZ_Suchworte_Systematik_und_Berufe_gueltig.xlsx')
    return data_kldb

def occupations_fur_berufsausbildung():
    '''
    The persons with berufsausbildung are only qualified for the occupations with the KldB codes which has 1, 2 at the end
    This function removes all the KldB codes other than 1, 2
    '''
    data_kldb = pd.read_excel('/home/gshaik@forschungsnetz.local/project/input/DKZ_Suchworte_Systematik_und_Berufe_gueltig.xlsx')
    df_filtered = data_kldb[~data_kldb["KldB-2010_(5-Steller)"].astype(str).str[-1].isin(["3", "4"])]
    return df_filtered

def Title_occupation(kldb_code):
    data_kldb = kldb_occupations()
    return list(data_kldb[data_kldb['KldB-2010_(5-Steller)']==kldb_code]['Berufsbenennungen'])



if __name__ == "__main__":
    kldb = 23224
    data_kldb_code = pd.read_excel('/home/gshaik@forschungsnetz.local/project/input/DKZ_Suchworte_Systematik_und_Berufe_gueltig.xlsx')
    print(list(data_kldb_code[data_kldb_code['KldB-2010_(5-Steller)']==23224]['Berufsbenennungen']))