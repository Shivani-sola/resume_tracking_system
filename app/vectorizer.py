# placeholder
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def train_models(texts, tfidf_path, svd_path, n_components=300):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(texts)
    # If not enough samples, use min(X.shape[0], n_components)
    n_comp = min(X.shape[0], n_components)
    svd = TruncatedSVD(n_components=n_comp)
    svd.fit(X)
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(svd, svd_path)

def load_models(tfidf_path, svd_path, n_components=300):
    tfidf = joblib.load(tfidf_path)
    svd = joblib.load(svd_path)
    def transform(texts):
        X = tfidf.transform(texts)
        vecs = svd.transform(X)
        # Pad with zeros if needed
        if vecs.shape[1] < n_components:
            import numpy as np
            pad_width = n_components - vecs.shape[1]
            vecs = np.pad(vecs, ((0,0),(0,pad_width)), 'constant')
        return vecs
    return tfidf, svd, transform
