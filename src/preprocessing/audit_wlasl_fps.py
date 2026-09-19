from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = (
    ROOT
    / "data"
    / "asl"
    / "WLASL100.json"
)

RAW_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "raw_videos_mp4"
)

OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "wlasl100_fps_audit.csv"
)


# ============================================================
# CHARGER WLASL100
# ============================================================

with open(
    JSON_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


rows = []


# ============================================================
# ANALYSER LES INSTANCES YOUTUBE
# ============================================================

for item in data:

    gloss = item["gloss"]

    for inst in item["instances"]:

        url = inst.get("url", "")

        if (
            "youtube.com" not in url
            and "youtu.be" not in url
        ):
            continue


        video_id = str(
            inst["video_id"]
        ).zfill(5)


        # ----------------------------------------------------
        # Youtube ID
        # ----------------------------------------------------

        parsed = urlparse(url)

        if "youtu.be" in url:

            youtube_id = (
                parsed.path
                .strip("/")
            )

        else:

            youtube_id = parse_qs(
                parsed.query
            ).get(
                "v",
                [None]
            )[0]


        metadata_fps = float(
            inst.get("fps", 25)
        )


        row = {

            "video":
                f"{video_id}.mp4",

            "video_id":
                video_id,

            "gloss":
                gloss,

            "source":
                inst.get("source"),

            "youtube_id":
                youtube_id,

            "metadata_fps":
                metadata_fps,

            "actual_fps":
                None,

            "fps_difference":
                None,

            "fps_mismatch":
                False,

            "frame_start":
                inst.get("frame_start"),

            "frame_end":
                inst.get("frame_end"),

            "raw_exists":
                False
        }


        if youtube_id is None:

            rows.append(row)
            continue


        raw_path = (
            RAW_DIR
            / f"{youtube_id}.mp4"
        )


        if not raw_path.exists():

            rows.append(row)
            continue


        row["raw_exists"] = True


        cap = cv2.VideoCapture(
            str(raw_path)
        )


        if not cap.isOpened():

            cap.release()

            rows.append(row)
            continue


        actual_fps = cap.get(
            cv2.CAP_PROP_FPS
        )


        cap.release()


        row["actual_fps"] = (
            actual_fps
        )


        difference = (
            actual_fps
            - metadata_fps
        )


        row["fps_difference"] = (
            difference
        )


        # Ce seuil sert uniquement à détecter
        # un vrai changement de FPS.
        row["fps_mismatch"] = (
            abs(difference) > 0.5
        )


        rows.append(row)


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# RÉSULTATS
# ============================================================

print("=" * 70)
print("AUDIT FPS WLASL100")
print("=" * 70)


print(
    "Instances YouTube :",
    len(df)
)


print(
    "Sources RAW retrouvées :",
    int(
        df["raw_exists"].sum()
    )
)


print(
    "Sources RAW manquantes :",
    int(
        (~df["raw_exists"]).sum()
    )
)


available = df[
    df["actual_fps"].notna()
].copy()


print(
    "\nInstances avec FPS analysable :",
    len(available)
)


print(
    "FPS identique/compatible :",
    int(
        (
            ~available["fps_mismatch"]
        ).sum()
    )
)


print(
    "FPS différent :",
    int(
        available["fps_mismatch"].sum()
    )
)


print(
    "\n========== FPS RÉELS =========="
)

print(
    available[
        "actual_fps"
    ]
    .round(3)
    .value_counts()
    .sort_index()
)


print(
    "\n========== MISMATCH PAR SOURCE =========="
)


mismatch = available[
    available["fps_mismatch"]
].copy()


if len(mismatch) == 0:

    print(
        "Aucun mismatch détecté."
    )

else:

    print(
        mismatch[
            "source"
        ]
        .value_counts(
            dropna=False
        )
    )


print(
    "\n========== RÉSUMÉ PAR SOURCE =========="
)


summary = (
    available
    .groupby(
        "source",
        dropna=False
    )
    .agg(
        instances=("video", "count"),
        mismatches=("fps_mismatch", "sum"),
        metadata_fps_min=("metadata_fps", "min"),
        metadata_fps_max=("metadata_fps", "max"),
        actual_fps_min=("actual_fps", "min"),
        actual_fps_max=("actual_fps", "max")
    )
    .sort_values(
        "mismatches",
        ascending=False
    )
)


print(
    summary.to_string()
)


print(
    "\n========== EXEMPLES DE MISMATCH =========="
)


print(
    mismatch[
        [
            "video",
            "gloss",
            "source",
            "metadata_fps",
            "actual_fps",
            "frame_start",
            "frame_end"
        ]
    ]
    .head(40)
    .to_string(
        index=False
    )
)


print("\nFichier créé :")
print(OUTPUT)