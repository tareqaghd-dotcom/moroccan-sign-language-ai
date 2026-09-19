from pathlib import Path
import json
import shutil

import pandas as pd


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CURRENT_CLEAN = (
    ROOT
    / "data"
    / "asl"
    / "asl_clean_final.csv"
)

CANDIDATE_REPORT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "combined_quality_report.csv"
)

FULL_WLASL = (
    ROOT
    / "data"
    / "asl"
    / "WLASL_v0.3.json"
)


CURRENT_VIDEOS_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "videos"
)

CANDIDATE_VIDEOS_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "candidate_batch1_processed"
)


FINAL_VIDEOS_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "final_wlasl100_videos"
)


OUTPUT_CSV = (
    ROOT
    / "data"
    / "asl"
    / "final_wlasl100.csv"
)

OUTPUT_CLASSES = (
    ROOT
    / "data"
    / "asl"
    / "final_wlasl100_classes.csv"
)


# ============================================================
# CLASSES
# ============================================================

REMOVE_CLASSES = {
    "TELL",
    "LAST",
    "HELP",
    "CHEAT",
    "MEDICINE",
    "TABLE"
}


ADD_CLASSES = {
    "AGAIN",
    "DOOR",
    "HAIR",
    "BROTHER",
    "GOOD",
    "COFFEE"
}


print("=" * 80)
print("CRÉATION DU WLASL100 FINAL")
print("=" * 80)


print(
    "Classes retirées :",
    sorted(REMOVE_CLASSES)
)

print(
    "Classes ajoutées :",
    sorted(ADD_CLASSES)
)


# ============================================================
# CHARGER LES MÉTADONNÉES WLASL COMPLÈTES
# ============================================================

with open(
    FULL_WLASL,
    "r",
    encoding="utf-8"
) as f:

    full_data = json.load(f)


metadata = {}


for item in full_data:

    gloss = str(
        item["gloss"]
    ).upper()


    for inst in item["instances"]:

        video_id = str(
            inst["video_id"]
        ).zfill(5)


        key = (
            video_id,
            gloss
        )


        metadata[key] = {

            "signer_id":
                inst.get(
                    "signer_id"
                ),

            "source":
                inst.get(
                    "source"
                ),

            "original_split":
                inst.get(
                    "split"
                ),

            "frame_start":
                inst.get(
                    "frame_start"
                ),

            "frame_end":
                inst.get(
                    "frame_end"
                ),

            "metadata_fps":
                inst.get(
                    "fps"
                )
        }


# ============================================================
# 1. DATASET ACTUEL
# ============================================================

current = pd.read_csv(
    CURRENT_CLEAN,
    dtype={"video": str}
)


current["gloss"] = (
    current["gloss"]
    .astype(str)
    .str.upper()
)


print(
    "\nDataset actuel :",
    len(current),
    "vidéos"
)

print(
    "Classes actuelles :",
    current["gloss"].nunique()
)


# Supprimer les 6 classes faibles
current_final = current[
    ~current[
        "gloss"
    ].isin(
        REMOVE_CLASSES
    )
].copy()


current_final[
    "origin"
] = "CURRENT"


print(
    "\nAprès suppression des 6 classes :"
)

print(
    "Vidéos :",
    len(current_final)
)

print(
    "Classes :",
    current_final[
        "gloss"
    ].nunique()
)


# ============================================================
# ENRICHIR LES MÉTADONNÉES ACTUELLES
# ============================================================

current_rows = []


for _, row in current_final.iterrows():

    video = str(
        row["video"]
    )

    video_id = (
        Path(video)
        .stem
        .zfill(5)
    )

    gloss = str(
        row["gloss"]
    ).upper()


    meta = metadata.get(
        (
            video_id,
            gloss
        ),
        {}
    )


    current_rows.append({

        "video":
            f"{video_id}.mp4",

        "gloss":
            gloss,

        "signer_id":
            meta.get(
                "signer_id"
            ),

        "source":
            meta.get(
                "source"
            ),

        "original_split":
            meta.get(
                "original_split"
            ),

        "frame_start":
            meta.get(
                "frame_start"
            ),

        "frame_end":
            meta.get(
                "frame_end"
            ),

        "metadata_fps":
            meta.get(
                "metadata_fps"
            ),

        "origin":
            "CURRENT"
    })


current_final = pd.DataFrame(
    current_rows
)


# ============================================================
# 2. NOUVELLES CLASSES
# ============================================================

candidate = pd.read_csv(
    CANDIDATE_REPORT,
    dtype={"video": str}
)


candidate["gloss"] = (
    candidate["gloss"]
    .astype(str)
    .str.upper()
)


candidate_final = candidate[
    (
        candidate[
            "gloss"
        ].isin(
            ADD_CLASSES
        )
    )
    &
    (
        candidate[
            "quality_decision"
        ]
        == "AUTO_KEEP"
    )
].copy()


candidate_rows = []


for _, row in candidate_final.iterrows():

    video = str(
        row["video"]
    )

    video_id = (
        Path(video)
        .stem
        .zfill(5)
    )

    gloss = str(
        row["gloss"]
    ).upper()


    meta = metadata.get(
        (
            video_id,
            gloss
        ),
        {}
    )


    candidate_rows.append({

        "video":
            f"{video_id}.mp4",

        "gloss":
            gloss,

        "signer_id":
            (
                row.get(
                    "signer_id"
                )
                if pd.notna(
                    row.get(
                        "signer_id"
                    )
                )
                else meta.get(
                    "signer_id"
                )
            ),

        "source":
            (
                row.get(
                    "source"
                )
                if pd.notna(
                    row.get(
                        "source"
                    )
                )
                else meta.get(
                    "source"
                )
            ),

        "original_split":
            meta.get(
                "original_split"
            ),

        "frame_start":
            meta.get(
                "frame_start"
            ),

        "frame_end":
            meta.get(
                "frame_end"
            ),

        "metadata_fps":
            meta.get(
                "metadata_fps"
            ),

        "origin":
            "NEW_CANDIDATE"
    })


candidate_final = pd.DataFrame(
    candidate_rows
)


print(
    "\nNouvelles vidéos ajoutées :",
    len(candidate_final)
)

print(
    "Nouvelles classes :",
    candidate_final[
        "gloss"
    ].nunique()
)


print(
    "\nRépartition nouvelles classes :"
)

print(
    candidate_final[
        "gloss"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 3. UNION
# ============================================================

final = pd.concat(
    [
        current_final,
        candidate_final
    ],
    ignore_index=True
)


# ============================================================
# CONTRÔLES
# ============================================================

print("\n")
print("=" * 80)
print("VÉRIFICATIONS")
print("=" * 80)


print(
    "Vidéos finales :",
    len(final)
)

print(
    "Classes finales :",
    final[
        "gloss"
    ].nunique()
)


duplicates = (
    final[
        "video"
    ]
    .duplicated()
    .sum()
)


print(
    "IDs vidéos dupliqués :",
    duplicates
)


missing_signer = (
    final[
        "signer_id"
    ]
    .isna()
    .sum()
)


print(
    "Signer manquant :",
    missing_signer
)


# ============================================================
# VÉRIFIER EXACTEMENT 100 CLASSES
# ============================================================

if (
    final[
        "gloss"
    ].nunique()
    != 100
):

    raise RuntimeError(
        "ERREUR : le dataset final ne contient pas exactement 100 classes."
    )


if duplicates != 0:

    raise RuntimeError(
        "ERREUR : IDs vidéo dupliqués."
    )


if missing_signer != 0:

    raise RuntimeError(
        "ERREUR : certaines vidéos n'ont pas de signer_id."
    )


# ============================================================
# 4. COPIER LES VIDÉOS
# ============================================================

if FINAL_VIDEOS_DIR.exists():

    shutil.rmtree(
        FINAL_VIDEOS_DIR
    )


FINAL_VIDEOS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


missing_files = []


for _, row in final.iterrows():

    video = row[
        "video"
    ]

    origin = row[
        "origin"
    ]


    if origin == "CURRENT":

        src = (
            CURRENT_VIDEOS_DIR
            / video
        )

    else:

        src = (
            CANDIDATE_VIDEOS_DIR
            / video
        )


    dst = (
        FINAL_VIDEOS_DIR
        / video
    )


    if not src.exists():

        missing_files.append(
            str(src)
        )

        continue


    shutil.copy2(
        src,
        dst
    )


# ============================================================
# FICHIERS MANQUANTS
# ============================================================

if missing_files:

    print(
        "\nERREUR : fichiers vidéo manquants :"
    )

    for path in missing_files:

        print(path)


    raise RuntimeError(
        f"{len(missing_files)} vidéos sont manquantes."
    )


# ============================================================
# 5. RAPPORT PAR CLASSE
# ============================================================

class_report = (
    final
    .groupby(
        "gloss"
    )
    .agg(
        videos=(
            "video",
            "count"
        ),

        signers=(
            "signer_id",
            "nunique"
        ),

        sources=(
            "source",
            "nunique"
        )
    )
    .reset_index()
)


class_report = (
    class_report
    .sort_values(
        [
            "videos",
            "signers"
        ],
        ascending=[
            True,
            True
        ]
    )
)


# ============================================================
# 6. SAUVEGARDE
# ============================================================

final.to_csv(
    OUTPUT_CSV,
    index=False
)


class_report.to_csv(
    OUTPUT_CLASSES,
    index=False
)


# ============================================================
# 7. RÉSULTAT FINAL
# ============================================================

print("\n")
print("=" * 80)
print("WLASL100 FINAL CRÉÉ")
print("=" * 80)


print(
    "Classes :",
    final[
        "gloss"
    ].nunique()
)


print(
    "Vidéos :",
    len(final)
)


print(
    "Signers uniques globaux :",
    final[
        "signer_id"
    ].nunique()
)


print(
    "\nNombre de vidéos par classe :"
)

print(
    class_report[
        "videos"
    ]
    .describe()
)


print(
    "\nNombre de signers par classe :"
)

print(
    class_report[
        "signers"
    ]
    .describe()
)


print(
    "\n========== 20 CLASSES LES PLUS PETITES =========="
)

print(
    class_report
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nCSV final :"
)

print(
    OUTPUT_CSV
)


print(
    "\nRapport classes :"
)

print(
    OUTPUT_CLASSES
)


print(
    "\nDossier vidéos final :"
)

print(
    FINAL_VIDEOS_DIR
)
