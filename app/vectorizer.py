# placeholder
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def train_models(texts, tfidf_path, svd_path):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=300)
    svd.fit(X)
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(svd, svd_path)

def load_models(tfidf_path, svd_path):
    return joblib.load(tfidf_path), joblib.load(svd_path)
