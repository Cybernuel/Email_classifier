# src/preprocess.py
"""
Simple preprocessing utilities for the Phishing Email Classifier.
- load_dataset: reads CSV and returns pandas DataFrame
- extract_urls: finds HTTP/HTTPS URLs in text
- featurize_row: builds the "text" field for ML and a couple simple numeric features
- prepare_dataset: returns X (texts) and y (labels)
"""

import pandas as pd
import re
from urllib.parse import urlparse
from typing import List, Tuple

URL_REGEX = re.compile(r'https?://[^\s\'"<>]+', re.IGNORECASE)

def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV dataset. Expects columns: id,label,subject,body"""
    df = pd.read_csv(path)
    # ensure basic columns
    for c in ("subject", "body", "label"):
        if c not in df.columns:
            df[c] = ""
    return df

def extract_urls(text: str) -> List[str]:
    """Return list of URLs found in a text string."""
    if not isinstance(text, str):
        return []
    return URL_REGEX.findall(text)

def get_domain(url: str) -> str:
    try:
        p = urlparse(url)
        return p.netloc.lower()
    except Exception:
        return ""



def featurize_row(subject: str, body: str) -> dict:
    """
    Create features useful for a baseline model:
    - combined_text: subject + body (lowercased)
    - n_urls: number of URLs in the email
    - has_ip_in_url: boolean if any URL contains an IP address
    - suspicious_tld: boolean if any url uses uncommon TLDs (quick heuristic)
    """
    subj = str(subject) if subject is not None else ""
    body = str(body) if body is not None else ""
    combined = (subj + " " + body).strip().lower()

    urls = extract_urls(combined)
    n_urls = len(urls)

    has_ip = False
    suspicious_tld = False
    for u in urls:
        dom = get_domain(u)
        # contains digits like an IPv4: quick heuristic
        if re.search(r'\d+\.\d+\.\d+\.\d+', dom):
            has_ip = True
        # suspicious TLDs (toy list)
        if dom.endswith((".xyz", ".info", ".top", ".club", ".pw")):
            suspicious_tld = True

    return {
        "combined_text": combined,
        "n_urls": n_urls,
        "has_ip_in_url": int(has_ip),
        "suspicious_tld": int(suspicious_tld)
    }


def prepare_dataset(csv_path: str):
    """
    Load CSV and return:
    - X_texts: list of combined_text strings (for TF-IDF)
    - X_meta: DataFrame with numeric meta features (n_urls, has_ip_in_url, suspicious_tld)
    - y: labels (0 = legit, 1 = phishing)
    """
    df = load_dataset(csv_path)
    records = []
    for _, r in df.iterrows():
        feats = featurize_row(r.get("subject", ""), r.get("body", ""))
        records.append(feats)
    meta = pd.DataFrame(records)
    # labels: accept 'phish'/'phishing'/'1' as positive
    y = df['label'].astype(str).str.lower().map(lambda v: 1 if ("phish" in v or v.strip()=="1") else 0).values
    X_texts = meta['combined_text'].tolist()
    X_meta = meta[['n_urls','has_ip_in_url','suspicious_tld']]
    return X_texts, X_meta, y
