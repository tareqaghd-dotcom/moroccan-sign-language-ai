from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

QUALITY = (
    ROOT
    / "data"
    / "asl"
    / "automatic_quality_decision.csv"
)

OUTPUT_REPORT = (
    ROOT
    / "data"
    / "asl"
    / "quality_decision_final.csv"
)

OUTPUT_DATASET = (
    ROOT
    / "data"
    / "asl"
    / "asl_clean_final.csv"
)


# ============================================================
# CHARGER LES DÉCISIONS AUTOMATIQUES
# ============================================================

df = pd.read_csv(QUALITY)


# Par défaut :
# AUTO_KEEP -> KEEP
# REVIEW    -> à décider manuellement
df["final_decision"] = df[
    "automatic_decision"
].map({
    "AUTO_KEEP": "KEEP",
    "REVIEW": "REVIEW"
})


df["final_reason"] = df[
    "automatic_reason"
]


# ============================================================
# DÉCISION MANUELLE POUR BOWLING
# ============================================================

mask = (
    df["video"] == "07383.mp4"
)

df.loc[
    mask,
    "final_decision"
] = "KEEP"

df.loc[
    mask,
    "final_reason"
] = (
    "validation manuelle : vidéo sombre mais geste visible et exploitable"
)


# ============================================================
# VÉRIFICATION
# ============================================================

print("=" * 70)
print("DÉCISION FINALE QUALITÉ")
print("=" * 70)

print(
    df["final_decision"]
    .value_counts()
)


unresolved = df[
    df["final_decision"] == "REVIEW"
]

print(
    "\nDécisions encore non résolues :",
    len(unresolved)
)


# ============================================================
# RAPPORT COMPLET
# ============================================================

df.to_csv(
    OUTPUT_REPORT,
    index=False
)


# ============================================================
# DATASET FINAL
# ============================================================

clean = df[
    df["final_decision"] == "KEEP"
].copy()


dataset_columns = [
    "video",
    "gloss",
    "split"
]

clean[
    dataset_columns
].to_csv(
    OUTPUT_DATASET,
    index=False
)


# ============================================================
# STATISTIQUES
# ============================================================

print(
    "\nVidéos dans dataset final :",
    len(clean)
)

print(
    "Classes :",
    clean["gloss"].nunique()
)

print(
    "\nRépartition split :"
)

print(
    clean["split"]
    .value_counts()
)


print(
    "\nNombre de vidéos par classe :"
)

class_counts = (
    clean["gloss"]
    .value_counts()
)

print(
    class_counts.describe()
)


print("\nRapport final :")
print(OUTPUT_REPORT)

print("\nDataset final :")
print(OUTPUT_DATASET)