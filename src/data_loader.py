# src/data_loader.py

"""
data_loader.py

Module ini digunakan untuk memuat dataset dari direktori `data/` baik dari folder raw maupun processed.
"""
import pandas as pd
import os

def load_raw_data(filename, folder='../data/raw/'):
    """
    Memuat data CSV dari folder data/raw/.
    
    Parameters:
        filename (str): Nama file, contoh: 'Retail_Transactions_Dataset.csv'
        folder (str): Lokasi folder relatif (default: data/raw)
        
    Returns:
        pd.DataFrame: DataFrame hasil pembacaan file
    """
    path = os.path.join(folder, filename)
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"File '{path}' tidak ditemukan.")
        return None

def load_processed_data(filename, folder='../data/processed/'):
    """
    Memuat data CSV dari folder data/processed/.
    
    Parameters:
        filename (str): Nama file, contoh: 'transactions.csv'
        folder (str): Lokasi folder relatif (default: data/processed)
        
    Returns:
        pd.DataFrame: DataFrame hasil pembacaan file
    """
    path = os.path.join(folder, filename)
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"File '{path}' tidak ditemukan.")
        return None

# Contoh penggunaan
if __name__ == "__main__":
    df_raw = load_raw_data('Retail_Transactions_Dataset.csv')
    df_processed = load_processed_data('transactions.csv')
    
    if df_raw is not None:
        print("Raw data preview:")
        print(df_raw.head())

    if df_processed is not None:
        print("\nProcessed data preview:")
        print(df_processed.head())