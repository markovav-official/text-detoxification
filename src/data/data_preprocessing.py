import re

import nltk
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm


nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

regex_digit = re.compile(r'\d+')
regex_non_alpha = re.compile(r'[^a-z|\s]+')
regex_spaces = re.compile(r'\s+')


def preprocess_text(text: str):
    text = text.lower()
    text = regex_digit.sub('', text)
    text = regex_non_alpha.sub('', text)
    text = regex_spaces.sub(' ', text).strip()

    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [stemmer.stem(word) for word in text]

    return ' '.join(text)


def preprocess_text_parallel(text_data: np.ndarray, n_jobs=-1):
    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        result = parallel(delayed(preprocess_text)(text) for text in tqdm(text_data, desc="Processing"))

    return result


def transform_combined_to_classified(df: pd.DataFrame):
    toxic_df = df[['toxic-en']]
    toxic_df.columns = ['text']
    toxic_df['toxic'] = 1

    neutral_df = df[['neutral-en']]
    neutral_df.columns = ['text']
    neutral_df['toxic'] = 0

    df = pd.concat([toxic_df, neutral_df], ignore_index=True)
    df['text'] = preprocess_text_parallel(df['text'])
    df = df[df['text'].str.len() > 0]
    return df