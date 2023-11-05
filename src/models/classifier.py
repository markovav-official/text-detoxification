from ..data.data_preprocessing import preprocess_text_parallel
import numpy as np
import pickle


def classify_text(text: np.ndarray, model_path='../models/logistic-regression-toxicity-classifier'):
    clf = pickle.load(open(model_path + '/model.bin', 'rb'))
    vectorizer = pickle.load(open(model_path + '/vectorizer.bin', 'rb'))
    return clf.predict(vectorizer.transform(preprocess_text_parallel(text)))


def classify_single_text(text: str, model_path='../models/logistic-regression-toxicity-classifier'):
    return classify_text(np.array([text]), model_path)[0]
