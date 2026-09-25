import re, unicodedata
from typing import Dict

MULTILINGUAL_LEGAL_SUFFIXES: Dict[str, str] = {
    "private": "pvt", "limited": "ltd", "incorporated": "inc", "corporation": "corp",
    "societe par actions simplifiee": "sas", "societe a responsabilite limitee": "sarl"
}

def normalize_multilingual_name(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [MULTILINGUAL_LEGAL_SUFFIXES.get(t, t) for t in text.split()]
    return " ".join(tokens)

def extract_international_postal(address: str, country: str = "") -> str:
    if not address: return ""
    matches = re.findall(r"\b\d{5,6}\b", str(address).upper())
    return matches[-1] if matches else ""