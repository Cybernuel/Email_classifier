# src/train.py
"""
Train script for baseline phishing classifier.
Creates:
- models/tfidf_vectorizer.joblib
- models/classifier.joblib
- models/meta_scaler.joblib
"""

import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from src.preprocess import prepare_dataset

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train(csv_path: str):
    X_texts, X_meta, y = prepare_dataset(csv_path)
    # TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=8000)
    X_tfidf = tfidf.fit_transform(X_texts)  # sparse

    # scale meta features
    scaler = StandardScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)

    # combine TF-IDF + meta by horizontally stacking arrays (convert meta to sparse then hstack)
    from scipy.sparse import hstack
    from scipy import sparse
    X_meta_sparse = sparse.csr_matrix(X_meta_scaled)
    X_combined = hstack([X_tfidf, X_meta_sparse])

    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # eval
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    print("Classification report:")
    print(classification_report(y_test, preds))
    try:
        print("ROC AUC:", roc_auc_score(y_test, probs))
    except Exception:
        pass

    # save artifacts
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(scaler, MODEL_DIR / "meta_scaler.joblib")
    joblib.dump(clf, MODEL_DIR / "classifier.joblib")
    print("Saved model artifacts to", MODEL_DIR)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="sample_data/sample_emails.csv", help="Path to CSV dataset")
    args = parser.parse_args()
    train(args.csv)
