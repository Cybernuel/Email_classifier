# src/explain.py
"""
LIME-based explainability wrapper.
Given a text and meta features, it will return the top contributing words for the phish class.
"""

import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from scipy.sparse import hstack
from src.preprocess import featurize_row

MODEL_DIR = "models"

def load_artifacts():
    tfidf = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.joblib")
    scaler = joblib.load(f"{MODEL_DIR}/meta_scaler.joblib")
    clf = joblib.load(f"{MODEL_DIR}/classifier.joblib")
    return tfidf, scaler, clf

def explain_email(subject: str, body: str, top_n=6):
    tfidf, scaler, clf = load_artifacts()
    text = (subject or "") + " " + (body or "")
    # LIME expects a function that maps raw strings -> prediction probabilities
    def predict_proba(texts):
        # texts: list[str]
        X_tfidf = tfidf.transform(texts)
        # for meta features we can't vary e.g. url counts easily; use featurized baseline
        # get meta features from original email
        meta = featurize_row(subject, body)
        meta_arr = np.array([[meta["n_urls"], meta["has_ip_in_url"], meta["suspicious_tld"]]])
        meta_scaled = scaler.transform(meta_arr)
        meta_sparse = np.repeat(meta_scaled, len(texts), axis=0)
        # stack
        X_comb = hstack([X_tfidf, meta_sparse])
        # get probabilities
        try:
            probs = clf.predict_proba(X_comb)
        except Exception:
            # fallback: use decision_function
            df = clf.decision_function(X_comb)
            # convert to pseudo-prob
            probs = np.vstack([1 - (1/(1+np.exp(-df))), 1/(1+np.exp(-df))]).T
        return probs

    explainer = LimeTextExplainer(class_names=["legit","phish"])
    exp = explainer.explain_instance(text, predict_proba, num_features=top_n)
    return exp.as_list()
