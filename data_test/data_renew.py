import pandas as pd
import numpy as np
def reshape_df(df) :
    df = df.drop(index=df.index[0], columns='Unnamed: 10', errors='ignore')
    df = df.reset_index(drop=True)
    df.rename(columns={'# Time':'Time'},inplace=True)
    return df
data_ftp=pd.read_csv('/Users/apple/git/ROM_code/data_test/FTP.csv',sep=';',skiprows=4)
data_ftp_new=reshape_df(data_ftp)
#print(data_ftp.columns)
data_ftp_new.drop('Unnamed: 8', axis=1, inplace=True)
data_ftp_new.to_csv('/Users/apple/git/ROM_code/data_test/FTP_new.csv')

data_hwfet=pd.read_csv('/Users/apple/git/ROM_code/data_test/HWFET.csv',sep=';',skiprows=4)
data_hwfet_new=reshape_df(data_hwfet)
#print(data_hwfet.columns)
data_hwfet_new.drop('Unnamed: 8', axis=1, inplace=True)
data_hwfet_new.to_csv('/Users/apple/git/ROM_code/data_test/HWFET_new.csv')

data_sc03=pd.read_csv('/Users/apple/git/ROM_code/data_test/SC03.csv',sep=';',skiprows=4)
data_sc03_new=reshape_df(data_sc03)
#print(data_hwfet.columns)
data_sc03_new.drop('Unnamed: 8', axis=1, inplace=True)
data_sc03_new.to_csv('/Users/apple/git/ROM_code/data_test/SC03_new.csv')

data_us06=pd.read_csv('/Users/apple/git/ROM_code/data_test/US06.csv',sep=';',skiprows=4)
data_us06_new=reshape_df(data_us06)
#print(data_hwfet.columns)
data_us06_new.drop('Unnamed: 8', axis=1, inplace=True)
data_us06_new.to_csv('/Users/apple/git/ROM_code/data_test/US06_new.csv')

data_nedt=pd.read_csv('/Users/apple/git/ROM_code/data_test/NEDT.csv',sep=';',skiprows=4)
data_nedt_new=reshape_df(data_nedt)
print(data_hwfet.columns)
data_nedt_new.drop('Unnamed: 8', axis=1, inplace=True)
data_nedt_new.to_csv('/Users/apple/git/ROM_code/data_test/NEDT_new.csv')