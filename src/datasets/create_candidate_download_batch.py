from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SUMMARY = (
    ROOT
    / "data"
    / "asl"
    / "wlasl_candidate_pool_summary.csv"
)

AVAILABILITY = (
    ROOT
    / "data"
    / "asl"
    / "candidate_pool_availability.csv"
)

MISSING = (
    ROOT
    / "data"
    / "asl"
    / "candidate_pool_missing_sources.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PARAMÈTRE
# ============================================================

N_CLASSES = 50


# ============================================================
# 1. TOP 50 NOUVELLES CLASSES
# ============================================================

summary = pd.read_csv(SUMMARY)

new_classes = (
    summary[
        summary["category"] == "NEW_CANDIDATE"
    ]
    .sort_values(
        [
            "total_instances",
            "n_signers",
            "n_sources",
            "signer_dominance"
        ],
        ascending=[
            False,
            False,
            False,
            True
        ]
    )
    .head(N_CLASSES)
    .copy()
)


selected_glosses = set(
    new_classes["gloss"]
)


print("=" * 80)
print("BATCH 1 : TOP 50 NOUVELLES CLASSES")
print("=" * 80)

print(
    "Classes sélectionnées :",
    len(selected_glosses)
)


# ============================================================
# 2. TOUTES LES INSTANCES DE CES CLASSES
# ============================================================

availability = pd.read_csv(
    AVAILABILITY,
    dtype={"video": str}
)


batch = availability[
    (availability["category"] == "NEW_CANDIDATE")
    &
    (availability["gloss"].isin(selected_glosses))
].copy()


print(
    "Instances théoriques du batch :",
    len(batch)
)


print(
    "\nStatuts actuels :"
)

print(
    batch["status"]
    .value_counts()
)


# ============================================================
# 3. SOURCES MANQUANTES
# ============================================================

missing = pd.read_csv(
    MISSING,
    dtype={"video": str}
)


batch_missing = missing[
    (missing["category"] == "NEW_CANDIDATE")
    &
    (missing["gloss"].isin(selected_glosses))
].copy()


print(
    "\nInstances à récupérer :",
    len(batch_missing)
)


print(
    "\nTypes d'URL :"
)

print(
    batch_missing["url_type"]
    .value_counts()
)


# ============================================================
# 4. YOUTUBE
#
# Une même source peut produire plusieurs instances.
# On déduplique donc sur youtube_id.
# ============================================================

youtube = batch_missing[
    batch_missing["url_type"] == "YOUTUBE"
].copy()


youtube_unique = (
    youtube[
        youtube["youtube_id"].notna()
    ]
    .sort_values(
        [
            "gloss",
            "youtube_id"
        ]
    )
    .drop_duplicates(
        subset=["youtube_id"]
    )
)


# ============================================================
# 5. DIRECT MP4
# ============================================================

direct = batch_missing[
    batch_missing["url_type"] == "DIRECT_MP4"
].copy()


# ============================================================
# 6. AUTRES SOURCES
# ============================================================

other = batch_missing[
    ~batch_missing["url_type"].isin(
        [
            "YOUTUBE",
            "DIRECT_MP4"
        ]
    )
].copy()


# ============================================================
# 7. RAW DÉJÀ DISPONIBLES
# ============================================================

raw_available = batch[
    batch["status"] == "RAW_SOURCE_AVAILABLE"
].copy()


# ============================================================
# 8. SAUVEGARDE
# ============================================================

new_classes.to_csv(
    OUTPUT_DIR / "selected_classes.csv",
    index=False
)

batch.to_csv(
    OUTPUT_DIR / "all_instances.csv",
    index=False
)

raw_available.to_csv(
    OUTPUT_DIR / "raw_already_available.csv",
    index=False
)

batch_missing.to_csv(
    OUTPUT_DIR / "to_fetch.csv",
    index=False
)

youtube_unique.to_csv(
    OUTPUT_DIR / "youtube_sources_unique.csv",
    index=False
)

direct.to_csv(
    OUTPUT_DIR / "direct_mp4.csv",
    index=False
)

other.to_csv(
    OUTPUT_DIR / "other_sources.csv",
    index=False
)


# ============================================================
# 9. STATS PAR CLASSE
# ============================================================

class_stats = (
    batch
    .groupby("gloss")
    .agg(
        theoretical=("video", "count"),

        already_clean=(
            "already_clean",
            "sum"
        ),

        raw_available=(
            "raw_youtube_exists",
            "sum"
        ),

        signers=(
            "signer_id",
            "nunique"
        )
    )
)


missing_stats = (
    batch_missing
    .groupby("gloss")
    .agg(
        to_fetch=("video", "count")
    )
)


class_stats = (
    class_stats
    .join(
        missing_stats,
        how="left"
    )
    .fillna({
        "to_fetch": 0
    })
)


class_stats["to_fetch"] = (
    class_stats["to_fetch"]
    .astype(int)
)


class_stats = class_stats.sort_values(
    [
        "theoretical",
        "signers"
    ],
    ascending=[
        False,
        False
    ]
)


class_stats.to_csv(
    OUTPUT_DIR / "class_stats.csv"
)


# ============================================================
# 10. AFFICHAGE FINAL
# ============================================================

print("\n")
print("=" * 80)
print("RÉSUMÉ DU BATCH 1")
print("=" * 80)

print(
    "Classes :",
    len(new_classes)
)

print(
    "Instances théoriques :",
    len(batch)
)

print(
    "RAW déjà disponibles :",
    len(raw_available)
)

print(
    "Instances à récupérer :",
    len(batch_missing)
)

print(
    "\nDIRECT_MP4 :",
    len(direct)
)

print(
    "Instances YouTube :",
    len(youtube)
)

print(
    "Sources YouTube uniques :",
    len(youtube_unique)
)

print(
    "Autres sources :",
    len(other)
)


print(
    "\n========== PAR TYPE DE SOURCE =========="
)

print(
    batch_missing["source"]
    .value_counts()
)


print(
    "\n========== 20 PREMIÈRES CLASSES =========="
)

print(
    class_stats
    .head(20)
    .to_string()
)


print(
    "\nFichiers créés dans :"
)

print(OUTPUT_DIR)