import re
import unicodedata
import pandas as pd


def normalize_text(text):
    """Basic normalization: unicode, case, whitespace, punctuation."""
    if pd.isnull(text):
        return text

    text = str(text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)      # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def extract_postal_code(address):
    """Pull a likely postal/pin code from an address string."""
    if pd.isnull(address):
        return None
    match = re.search(r"\b\d{5,6}\b", str(address))
    return match.group() if match else None


def tokenize(text):
    if pd.isnull(text) or text == "":
        return []
    return text.split()


def clean_dataframe(df, name_col="business_name", addr_col="business_address"):
    """Add normalized, tokenized, and extracted columns while keeping raw ones."""
    df = df.copy()

    if name_col in df.columns:
        df["name_normalized"] = df[name_col].apply(normalize_text)
        df["name_tokens"] = df["name_normalized"].apply(tokenize)
        df["name_norm"] = df["name_normalized"].fillna("")
    else:
        print(f"Warning: name column '{name_col}' not found")

    if addr_col in df.columns:
        df["postal_code"] = df[addr_col].apply(extract_postal_code)
        df["address_normalized"] = df[addr_col].apply(normalize_text)
        df["address_tokens"] = df["address_normalized"].apply(tokenize)
        df["address_norm"] = df["address_normalized"].fillna("")
        df["postal"] = df["postal_code"].fillna("").astype(str)
    else:
        print(f"Warning: address column '{addr_col}' not found")

    return df