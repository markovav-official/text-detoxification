import os

import pandas as pd


def add_to_combined(dataset: pd.DataFrame, file: str = '../data/interim/combined.tsv', is_initial: bool = False):
    os.makedirs(os.path.dirname(file), exist_ok=True)
    if not os.path.exists(file) or is_initial:
        dataset.to_csv(file, sep='\t', index=False, header=True)
        return dataset

    df: pd.DataFrame = pd.read_csv(file, sep='\t', header=0)
    df = pd.concat([df, dataset], ignore_index=True)
    df.to_csv(file, sep='\t', index=False)
    return df
