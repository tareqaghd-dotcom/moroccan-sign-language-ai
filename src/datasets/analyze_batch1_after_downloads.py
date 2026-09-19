from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BATCH_DIR = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
)

ALL_INSTANCES = (
    BATCH_DIR
    / "all_instances.csv"
)

DIRECT_REPORT = (
    BATCH_DIR
    / "direct_mp4_download_report.csv"
)

YOUTUBE_REPORT = (
    BATCH_DIR
    / "youtube_download_report.csv"
)

OUTPUT_INSTANCES = (
    BATCH_DIR
    / "availability_after_downloads.csv"
)

OUTPUT_CLASSES = (
    BATCH_DIR
    / "class_availability_after_downloads.csv"
)


# ============================================================
# 1. CHARGER LES INSTANCES
# ============================================================

df = pd.read_csv(
    ALL_INSTANCES,
    dtype={
        "video": str,
        "youtube_id": str
    }
)


# Nettoyage des NaN convertis en chaînes
df["youtube_id"] = (
    df["youtube_id"]
    .replace(
        {
            "nan": None,
            "None": None,
            "": None
        }
    )
)


# ============================================================
# 2. RAPPORT DIRECT MP4
# ============================================================

direct = pd.read_csv(
    DIRECT_REPORT,
    dtype={
        "video": str
    }
)


def as_bool(series):

    return (
        series.astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes"
            ]
        )
    )


direct["success_bool"] = (
    as_bool(
        direct["success"]
    )
)


direct_success_videos = set(
    direct.loc[
        direct["success_bool"],
        "video"
    ]
)


# ============================================================
# 3. RAPPORT YOUTUBE
# ============================================================

youtube = pd.read_csv(
    YOUTUBE_REPORT,
    dtype={
        "youtube_id": str
    }
)


youtube["success_bool"] = (
    as_bool(
        youtube["success"]
    )
)


youtube_success_ids = set(
    youtube.loc[
        youtube["success_bool"],
        "youtube_id"
    ]
)


# ============================================================
# 4. DISPONIBILITÉ DE CHAQUE INSTANCE
# ============================================================

df["direct_downloaded"] = (
    df["video"]
    .isin(
        direct_success_videos
    )
)


df["youtube_downloaded"] = (
    df["youtube_id"]
    .isin(
        youtube_success_ids
    )
)


# RAW présent avant nos téléchargements
df["raw_preexisting"] = (
    df["status"]
    == "RAW_SOURCE_AVAILABLE"
)


# L'instance est maintenant potentiellement exploitable si :
# - c'est un clip direct téléchargé
# - OU sa source YouTube vient d'être téléchargée
# - OU la source RAW existait déjà
df["source_available_now"] = (
    df["direct_downloaded"]
    |
    df["youtube_downloaded"]
    |
    df["raw_preexisting"]
)


# ============================================================
# 5. TYPE D'ACCÈS
# ============================================================

def access_type(row):

    if row["direct_downloaded"]:
        return "DIRECT_CLIP"

    if row["youtube_downloaded"]:
        return "YOUTUBE_NEW"

    if row["raw_preexisting"]:
        return "RAW_PREEXISTING"

    return "UNAVAILABLE"


df["access_type"] = (
    df.apply(
        access_type,
        axis=1
    )
)


# ============================================================
# 6. STATISTIQUES GLOBALES
# ============================================================

print("=" * 80)
print("BATCH 1 APRÈS LES TÉLÉCHARGEMENTS")
print("=" * 80)

print(
    "Instances théoriques :",
    len(df)
)

print(
    "Instances maintenant accessibles :",
    int(
        df["source_available_now"]
        .sum()
    )
)

print(
    "Instances encore indisponibles :",
    int(
        (
            ~df["source_available_now"]
        ).sum()
    )
)


print(
    "\n========== TYPE D'ACCÈS =========="
)

print(
    df["access_type"]
    .value_counts()
)


# ============================================================
# 7. RAPPORT PAR CLASSE
# ============================================================

rows = []


for gloss, group in df.groupby(
    "gloss"
):

    available = group[
        group[
            "source_available_now"
        ]
    ]


    total = len(
        group
    )


    n_available = len(
        available
    )


    n_signers_available = (
        available[
            "signer_id"
        ]
        .nunique()
    )


    if len(available) > 0:

        signer_counts = (
            available[
                "signer_id"
            ]
            .value_counts()
        )

        max_same_signer = int(
            signer_counts.iloc[0]
        )

        dominance = (
            max_same_signer
            / n_available
        )

    else:

        max_same_signer = 0
        dominance = 0


    rows.append({

        "gloss":
            gloss,

        "theoretical":
            total,

        "available_now":
            n_available,

        "missing_now":
            total - n_available,

        "available_ratio":
            n_available / total
            if total > 0
            else 0,

        "signers_available":
            n_signers_available,

        "max_same_signer":
            max_same_signer,

        "signer_dominance":
            dominance,

        "direct_clips":
            int(
                (
                    available[
                        "access_type"
                    ]
                    == "DIRECT_CLIP"
                ).sum()
            ),

        "youtube_new":
            int(
                (
                    available[
                        "access_type"
                    ]
                    == "YOUTUBE_NEW"
                ).sum()
            ),

        "raw_preexisting":
            int(
                (
                    available[
                        "access_type"
                    ]
                    == "RAW_PREEXISTING"
                ).sum()
            )
    })


classes = pd.DataFrame(
    rows
)


classes = classes.sort_values(
    [
        "available_now",
        "signers_available",
        "signer_dominance"
    ],
    ascending=[
        False,
        False,
        True
    ]
)


# ============================================================
# 8. SAUVEGARDE
# ============================================================

df.to_csv(
    OUTPUT_INSTANCES,
    index=False
)

classes.to_csv(
    OUTPUT_CLASSES,
    index=False
)


# ============================================================
# 9. RÉSUMÉ
# ============================================================

print(
    "\n========== DISPONIBILITÉ PAR CLASSE =========="
)


print(
    "Classes avec >= 12 instances accessibles :",
    int(
        (
            classes[
                "available_now"
            ]
            >= 12
        ).sum()
    )
)


print(
    "Classes avec >= 10 instances accessibles :",
    int(
        (
            classes[
                "available_now"
            ]
            >= 10
        ).sum()
    )
)


print(
    "Classes avec >= 8 instances accessibles :",
    int(
        (
            classes[
                "available_now"
            ]
            >= 8
        ).sum()
    )
)


print(
    "Classes avec < 8 instances accessibles :",
    int(
        (
            classes[
                "available_now"
            ]
            < 8
        ).sum()
    )
)


print(
    "\n========== 50 CLASSES =========="
)

print(
    classes[
        [
            "gloss",
            "theoretical",
            "available_now",
            "signers_available",
            "direct_clips",
            "youtube_new",
            "raw_preexisting",
            "missing_now",
            "signer_dominance"
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nRapport instances :"
)

print(
    OUTPUT_INSTANCES
)


print(
    "\nRapport classes :"
)

print(
    OUTPUT_CLASSES
)
