import pandas as pd

def read_raw_data():
    return pd.read_csv('../data/raw/filtered.tsv', sep='\t', index_col=0)
