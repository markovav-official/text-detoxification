import re

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

regex_digit = re.compile(r'\d+')
regex_non_alpha = re.compile(r'[^a-z|\s]+')
regex_spaces = re.compile(r'\s+')


def preprocess_text(text):
    text = text.lower()
    text = regex_digit.sub('', text)
    text = regex_non_alpha.sub('', text)
    text = regex_spaces.sub(' ', text).strip()

    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [stemmer.stem(word) for word in text]

    return ' '.join(text)


def preprocess_text_parallel(df: pd.DataFrame, n_jobs=-1):
    text_data = df['text'].to_numpy()

    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        result = parallel(delayed(preprocess_text)(text) for text in tqdm(text_data, desc="Processing"))

    df['text'] = result
    return df[df['text'].str.len() > 0]
