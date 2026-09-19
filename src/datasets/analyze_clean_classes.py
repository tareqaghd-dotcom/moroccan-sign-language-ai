from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CLEAN_CSV = (
    ROOT
    / "data"
    / "asl"
    / "asl_clean_final.csv"
)

WLASL_JSON = (
    ROOT
    / "data"
    / "asl"
    / "WLASL100.json"
)

OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "class_quality_report.csv"
)


# ============================================================
# 1. DATASET PROPRE
# ============================================================

clean = pd.read_csv(CLEAN_CSV)

clean["video_id"] = (
    clean["video"]
    .str.replace(".mp4", "", regex=False)
    .astype(str)
    .str.zfill(5)
)


# ============================================================
# 2. MÉTADONNÉES WLASL
# ============================================================

with open(
    WLASL_JSON,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


metadata = []


for item in data:

    gloss = item["gloss"]

    for inst in item["instances"]:

        video_id = str(
            inst["video_id"]
        ).zfill(5)

        metadata.append({
            "video_id": video_id,
            "gloss_metadata": gloss,
            "signer_id": inst.get("signer_id"),
            "source": inst.get("source"),
            "original_split": inst.get("split")
        })


metadata = pd.DataFrame(metadata)


# ============================================================
# 3. MERGE
# ============================================================

df = clean.merge(
    metadata,
    on="video_id",
    how="left"
)


print("=" * 80)
print("VÉRIFICATION MÉTADONNÉES")
print("=" * 80)

print(
    "Vidéos propres :",
    len(clean)
)

print(
    "Vidéos avec signer connu :",
    df["signer_id"].notna().sum()
)

print(
    "Vidéos sans signer :",
    df["signer_id"].isna().sum()
)


# ============================================================
# 4. STATISTIQUES PAR CLASSE
# ============================================================

rows = []


for gloss, group in df.groupby("gloss"):

    total = len(group)

    train = (
        group["split"]
        .eq("train")
        .sum()
    )

    val = (
        group["split"]
        .eq("val")
        .sum()
    )

    test = (
        group["split"]
        .eq("test")
        .sum()
    )


    signer_counts = (
        group["signer_id"]
        .dropna()
        .value_counts()
    )


    n_signers = (
        group["signer_id"]
        .dropna()
        .nunique()
    )


    if len(signer_counts) > 0:

        max_signer_videos = int(
            signer_counts.iloc[0]
        )

        signer_dominance = (
            max_signer_videos
            / total
        )

    else:

        max_signer_videos = 0
        signer_dominance = 0


    n_sources = (
        group["source"]
        .dropna()
        .nunique()
    )


    rows.append({

        "gloss": gloss,

        "total_clean": total,

        "train": train,
        "val": val,
        "test": test,

        "n_signers": n_signers,

        "max_videos_same_signer":
            max_signer_videos,

        "signer_dominance":
            signer_dominance,

        "n_sources":
            n_sources,

        "missing_val":
            val == 0,

        "missing_test":
            test == 0,

        "missing_train":
            train == 0
    })


report = pd.DataFrame(rows)


# ============================================================
# 5. INDICATEURS DE FRAGILITÉ
#
# Ce n'est PAS encore une décision de suppression.
# ============================================================

report["few_videos"] = (
    report["total_clean"] < 8
)

report["few_signers"] = (
    report["n_signers"] < 3
)

report["high_signer_dominance"] = (
    report["signer_dominance"] > 0.50
)


report["issue_count"] = (
    report[
        [
            "few_videos",
            "few_signers",
            "high_signer_dominance",
            "missing_train",
            "missing_val",
            "missing_test"
        ]
    ]
    .sum(axis=1)
)


# ============================================================
# 6. SAUVEGARDE
# ============================================================

report = report.sort_values(
    [
        "issue_count",
        "total_clean",
        "n_signers"
    ],
    ascending=[
        False,
        True,
        True
    ]
)


report.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# 7. AFFICHAGE
# ============================================================

print("\n")
print("=" * 80)
print("ANALYSE DES 100 CLASSES")
print("=" * 80)

print(
    "Classes :",
    len(report)
)

print(
    "Classes avec < 8 vidéos :",
    int(report["few_videos"].sum())
)

print(
    "Classes avec < 3 signers :",
    int(report["few_signers"].sum())
)

print(
    "Classes avec > 50 % provenant du même signer :",
    int(
        report[
            "high_signer_dominance"
        ].sum()
    )
)

print(
    "Classes sans validation :",
    int(report["missing_val"].sum())
)

print(
    "Classes sans test :",
    int(report["missing_test"].sum())
)


print(
    "\nNombre de vidéos propres par classe :"
)

print(
    report["total_clean"]
    .describe()
)


print(
    "\nNombre de signers par classe :"
)

print(
    report["n_signers"]
    .describe()
)


print(
    "\n========== 25 CLASSES À ÉTUDIER EN PRIORITÉ =========="
)

cols = [
    "gloss",
    "total_clean",
    "train",
    "val",
    "test",
    "n_signers",
    "max_videos_same_signer",
    "signer_dominance",
    "n_sources",
    "issue_count"
]

print(
    report[
        cols
    ]
    .head(25)
    .to_string(index=False)
)


print("\nRapport créé :")
print(OUTPUT)