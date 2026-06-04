from scipy.sparse import load_npz
import joblib
import numpy as np

def load_data(vectorizer_path, x_train_path, x_test_path, y_train_path, y_test_path):
    vectorizer = joblib.load(vectorizer_path) # needed when model predicts new sentences

    X_train_tfidf = load_npz(x_train_path)
    X_test_tfidf = load_npz(x_test_path)

    y_train = np.load(y_train_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)

    return vectorizer, X_train_tfidf, X_test_tfidf, y_train, y_test

