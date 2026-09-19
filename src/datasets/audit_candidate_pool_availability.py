from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

POOL_JSON = (
    ROOT
    / "data"
    / "asl"
    / "wlasl_candidate_pool.json"
)

POOL_SUMMARY = (
    ROOT
    / "data"
    / "asl"
    / "wlasl_candidate_pool_summary.csv"
)

CURRENT_CLEAN = (
    ROOT
    / "data"
    / "asl"
    / "asl_clean_final.csv"
)

VIDEOS_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "videos"
)

RAW_YOUTUBE_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "raw_videos_mp4"
)

OUTPUT_INSTANCES = (
    ROOT
    / "data"
    / "asl"
    / "candidate_pool_availability.csv"
)

OUTPUT_CLASSES = (
    ROOT
    / "data"
    / "asl"
    / "candidate_pool_class_availability.csv"
)


# ============================================================
# YOUTUBE ID
# ============================================================

def get_youtube_id(url):

    if not isinstance(url, str):
        return None

    try:

        parsed = urlparse(url)

        if "youtu.be" in parsed.netloc:
            return parsed.path.strip("/").split("/")[0]

        if (
            "youtube.com" in parsed.netloc
            or "www.youtube.com" in parsed.netloc
        ):

            video_id = parse_qs(
                parsed.query
            ).get("v", [None])[0]

            return video_id

    except Exception:
        pass

    return None


# ============================================================
# CHARGEMENT
# ============================================================

with open(
    POOL_JSON,
    "r",
    encoding="utf-8"
) as f:
    pool = json.load(f)


summary = pd.read_csv(
    POOL_SUMMARY
)

summary["gloss_lower"] = (
    summary["gloss"]
    .str.lower()
)


category_map = dict(
    zip(
        summary["gloss_lower"],
        summary["category"]
    )
)


current_clean = pd.read_csv(
    CURRENT_CLEAN,
    dtype={"video": str}
)

clean_videos = set(
    current_clean["video"]
)


# ============================================================
# FLATTEN DES 4411 INSTANCES
# ============================================================

rows = []


for item in pool:

    gloss = item["gloss"]

    category = category_map.get(
        gloss.lower(),
        "UNKNOWN"
    )

    for inst in item["instances"]:

        video_id = str(
            inst["video_id"]
        ).zfill(5)

        video_name = (
            f"{video_id}.mp4"
        )

        url = inst.get(
            "url",
            ""
        )

        youtube_id = get_youtube_id(
            url
        )


        processed_path = (
            VIDEOS_DIR
            / video_name
        )

        processed_exists = (
            processed_path.exists()
        )


        if processed_exists:

            processed_size = (
                processed_path
                .stat()
                .st_size
            )

        else:

            processed_size = 0


        # Une vidéo de 114 octets n'est évidemment pas exploitable.
        processed_nonempty = (
            processed_exists
            and processed_size > 1024
        )


        already_clean = (
            video_name
            in clean_videos
        )


        # ----------------------------------------------------
        # Source YouTube RAW déjà disponible ?
        # ----------------------------------------------------

        raw_youtube_exists = False
        raw_youtube_path = None


        if youtube_id:

            raw_youtube_path = (
                RAW_YOUTUBE_DIR
                / f"{youtube_id}.mp4"
            )

            raw_youtube_exists = (
                raw_youtube_path.exists()
                and raw_youtube_path.stat().st_size > 1024
            )


        # ----------------------------------------------------
        # Statut
        # ----------------------------------------------------

        if already_clean:

            status = "ALREADY_CLEAN"

        elif processed_nonempty:

            status = "PROCESSED_EXISTS_NOT_VALIDATED"

        elif raw_youtube_exists:

            status = "RAW_SOURCE_AVAILABLE"

        else:

            status = "SOURCE_TO_FETCH"


        rows.append({

            "video":
                video_name,

            "video_id":
                video_id,

            "gloss":
                gloss,

            "category":
                category,

            "split":
                inst.get("split"),

            "source":
                inst.get("source"),

            "signer_id":
                inst.get("signer_id"),

            "frame_start":
                inst.get("frame_start"),

            "frame_end":
                inst.get("frame_end"),

            "metadata_fps":
                inst.get("fps"),

            "url":
                url,

            "youtube_id":
                youtube_id,

            "already_clean":
                already_clean,

            "processed_exists":
                processed_exists,

            "processed_size":
                processed_size,

            "raw_youtube_exists":
                raw_youtube_exists,

            "status":
                status
        })


df = pd.DataFrame(
    rows
)


# ============================================================
# CONTRÔLES
# ============================================================

print("=" * 80)
print("AUDIT DU POOL DE 250 CLASSES")
print("=" * 80)

print(
    "Instances théoriques :",
    len(df)
)

print(
    "Classes :",
    df["gloss"].nunique()
)


print("\n========== STATUT DES INSTANCES ==========")

print(
    df["status"]
    .value_counts()
)


print(
    "\nDéjà propres :",
    int(
        df["already_clean"].sum()
    )
)


print(
    "Fichiers processed présents > 1 Ko :",
    int(
        df["processed_exists"]
        .where(
            df["processed_size"] > 1024,
            False
        )
        .sum()
    )
)


print(
    "\nInstances YouTube :",
    int(
        df["youtube_id"]
        .notna()
        .sum()
    )
)


print(
    "Instances YouTube dont le RAW est déjà présent :",
    int(
        df["raw_youtube_exists"]
        .sum()
    )
)


# ============================================================
# SOURCES YOUTUBE UNIQUES
# ============================================================

youtube = df[
    df["youtube_id"].notna()
].copy()


print(
    "\nSources YouTube uniques :",
    youtube["youtube_id"].nunique()
)


available_raw_sources = (
    youtube[
        youtube["raw_youtube_exists"]
    ]["youtube_id"]
    .nunique()
)


missing_raw_sources = (
    youtube[
        ~youtube["raw_youtube_exists"]
    ]["youtube_id"]
    .nunique()
)


print(
    "Sources YouTube RAW déjà présentes :",
    available_raw_sources
)

print(
    "Sources YouTube RAW manquantes :",
    missing_raw_sources
)


# ============================================================
# PAR CATÉGORIE
# ============================================================

print(
    "\n========== CURRENT VS NEW_CANDIDATE =========="
)


category_stats = (
    df
    .groupby("category")
    .agg(
        instances=("video", "count"),
        classes=("gloss", "nunique"),
        already_clean=("already_clean", "sum"),
        raw_available=("raw_youtube_exists", "sum")
    )
)


print(
    category_stats.to_string()
)


# ============================================================
# STATISTIQUES PAR CLASSE
# ============================================================

class_report = (
    df
    .groupby(
        ["gloss", "category"],
        as_index=False
    )
    .agg(
        theoretical_instances=(
            "video",
            "count"
        ),

        n_signers=(
            "signer_id",
            "nunique"
        ),

        already_clean=(
            "already_clean",
            "sum"
        ),

        processed_exists=(
            "processed_exists",
            "sum"
        ),

        raw_youtube_available=(
            "raw_youtube_exists",
            "sum"
        )
    )
)


class_report[
    "not_yet_clean"
] = (
    class_report[
        "theoretical_instances"
    ]
    - class_report[
        "already_clean"
    ]
)


# ============================================================
# SAUVEGARDE
# ============================================================

df.to_csv(
    OUTPUT_INSTANCES,
    index=False
)

class_report.to_csv(
    OUTPUT_CLASSES,
    index=False
)


# ============================================================
# NOUVELLES CLASSES
# ============================================================

new_classes = (
    class_report[
        class_report["category"]
        == "NEW_CANDIDATE"
    ]
    .sort_values(
        [
            "raw_youtube_available",
            "theoretical_instances",
            "n_signers"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
)


print(
    "\n========== 30 NOUVELLES CLASSES AVEC LE PLUS DE DONNÉES DÉJÀ ACCESSIBLES =========="
)

print(
    new_classes[
        [
            "gloss",
            "theoretical_instances",
            "n_signers",
            "already_clean",
            "processed_exists",
            "raw_youtube_available"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


print("\nRapport instances :")
print(OUTPUT_INSTANCES)

print("\nRapport classes :")
print(OUTPUT_CLASSES)