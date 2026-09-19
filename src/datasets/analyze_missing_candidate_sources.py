from pathlib import Path
from urllib.parse import urlparse
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_pool_availability.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_pool_missing_sources.csv"
)


df = pd.read_csv(INPUT)


# ============================================================
# GARDER UNIQUEMENT CE QUI MANQUE
# ============================================================

missing = df[
    df["status"] == "SOURCE_TO_FETCH"
].copy()


# ============================================================
# TYPE D'URL
# ============================================================

def classify_url(url):

    if not isinstance(url, str):
        return "UNKNOWN"

    url_lower = url.lower()

    if (
        "youtube.com" in url_lower
        or "youtu.be" in url_lower
    ):
        return "YOUTUBE"

    if url_lower.endswith(".mp4"):
        return "DIRECT_MP4"

    try:
        domain = (
            urlparse(url)
            .netloc
            .lower()
            .replace("www.", "")
        )

        if domain:
            return domain

    except Exception:
        pass

    return "OTHER"


missing["url_type"] = (
    missing["url"]
    .apply(classify_url)
)


# ============================================================
# DOMAIN
# ============================================================

def get_domain(url):

    if not isinstance(url, str):
        return ""

    try:
        return (
            urlparse(url)
            .netloc
            .lower()
            .replace("www.", "")
        )

    except Exception:
        return ""


missing["domain"] = (
    missing["url"]
    .apply(get_domain)
)


# ============================================================
# STATS GLOBALES
# ============================================================

print("=" * 80)
print("ANALYSE DES SOURCES MANQUANTES")
print("=" * 80)

print(
    "Instances SOURCE_TO_FETCH :",
    len(missing)
)

print(
    "Classes concernées :",
    missing["gloss"].nunique()
)

print(
    "\nURLs uniques :",
    missing["url"].nunique()
)


print(
    "\n========== PAR CATÉGORIE =========="
)

print(
    missing["category"]
    .value_counts()
)


print(
    "\n========== PAR TYPE D'URL =========="
)

print(
    missing["url_type"]
    .value_counts()
)


print(
    "\n========== PAR SOURCE WLASL =========="
)

print(
    missing["source"]
    .value_counts(dropna=False)
    .head(30)
)


print(
    "\n========== PAR DOMAINE =========="
)

print(
    missing["domain"]
    .value_counts()
    .head(30)
)


# ============================================================
# UNIQUEMENT NOUVELLES CLASSES
# ============================================================

new_missing = missing[
    missing["category"]
    == "NEW_CANDIDATE"
].copy()


print(
    "\n========== NOUVELLES CLASSES =========="
)

print(
    "Instances manquantes :",
    len(new_missing)
)

print(
    "Classes :",
    new_missing["gloss"].nunique()
)

print(
    "URLs uniques :",
    new_missing["url"].nunique()
)


print(
    "\nTypes d'URL :"
)

print(
    new_missing["url_type"]
    .value_counts()
)


# ============================================================
# MANQUANTS PAR CLASSE
# ============================================================

class_stats = (
    new_missing
    .groupby("gloss")
    .agg(
        missing_instances=("video", "count"),
        unique_urls=("url", "nunique"),
        signers=("signer_id", "nunique")
    )
    .sort_values(
        [
            "missing_instances",
            "signers"
        ],
        ascending=[
            False,
            False
        ]
    )
)


print(
    "\n========== 40 CLASSES AVEC LE PLUS DE DONNÉES À RÉCUPÉRER =========="
)

print(
    class_stats
    .head(40)
    .to_string()
)


# ============================================================
# YOUTUBE UNIQUES MANQUANTS
# ============================================================

youtube_missing = missing[
    missing["url_type"]
    == "YOUTUBE"
].copy()


print(
    "\n========== YOUTUBE =========="
)

print(
    "Instances YouTube manquantes :",
    len(youtube_missing)
)

print(
    "Sources YouTube uniques manquantes :",
    youtube_missing["youtube_id"]
    .dropna()
    .nunique()
)


# ============================================================
# SAUVEGARDE
# ============================================================

missing.to_csv(
    OUTPUT,
    index=False
)


print("\nRapport créé :")
print(OUTPUT)