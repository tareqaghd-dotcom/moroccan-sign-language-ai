from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

REPORT_CSV = (
    ROOT
    / "data"
    / "asl"
    / "combined_quality_report.csv"
)

VIDEOS_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "videos"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "quality_review"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PARAMÈTRES
# ============================================================

N_VIDEOS = 30
N_FRAMES = 12

FRAME_WIDTH = 320
FRAME_HEIGHT = 180


# ============================================================
# CHARGER LE RAPPORT
# ============================================================

df = pd.read_csv(REPORT_CSV)

df = df.head(N_VIDEOS)


print("Vidéos à examiner :", len(df))


# ============================================================
# EXTRAIRE DES FRAMES RÉPARTIES DANS LA VIDÉO
# ============================================================

def extract_frames(video_path, n_frames=12):

    cap = cv2.VideoCapture(str(video_path))

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frames = []

    if total <= 0:
        cap.release()
        return frames


    positions = np.linspace(
        0,
        total - 1,
        n_frames,
        dtype=int
    )


    for pos in positions:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(pos)
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            continue


        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )


        # numéro de frame
        cv2.putText(
            frame,
            f"frame {pos}",
            (8, 20),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,

            (255, 255, 255),
            1,

            cv2.LINE_AA
        )


        frames.append(frame)


    cap.release()

    return frames


# ============================================================
# CRÉER UNE PLANCHE POUR UNE VIDÉO
# ============================================================

def create_sheet(row):

    video_name = row["video"]

    path = VIDEOS_DIR / video_name

    frames = extract_frames(
        path,
        N_FRAMES
    )


    if not frames:

        print(
            "Impossible de lire :",
            video_name
        )

        return


    # compléter si moins de 12 frames extraites
    while len(frames) < N_FRAMES:

        blank = np.zeros(
            (
                FRAME_HEIGHT,
                FRAME_WIDTH,
                3
            ),
            dtype=np.uint8
        )

        frames.append(blank)


    # 4 images par ligne
    rows = []

    for i in range(
        0,
        N_FRAMES,
        4
    ):

        line = np.hstack(
            frames[i:i + 4]
        )

        rows.append(line)


    sheet = np.vstack(rows)


    # ========================================================
    # ZONE D'INFORMATIONS
    # ========================================================

    info_height = 130

    info = np.zeros(
        (
            info_height,
            sheet.shape[1],
            3
        ),
        dtype=np.uint8
    )


    gloss = (
        row["gloss_mp"]
        if "gloss_mp" in row.index
        else row.get("gloss", "")
    )


    split = (
        row["split_mp"]
        if "split_mp" in row.index
        else row.get("split", "")
    )


    texts = [

        f"{video_name} | gloss={gloss} | split={split}",

        (
            f"any_hand={row['any_hand_ratio']:.3f} | "
            f"middle_hand={row['middle_hand_ratio']:.3f} | "
            f"pose={row['pose_ratio']:.3f}"
        ),

        (
            f"long_no_hand={row['longest_no_hand_run_ratio']:.3f} | "
            f"visual_warnings={int(row['visual_warning_count'])} | "
            f"MP_flags={int(row['mediapipe_flag_count'])}"
        )
    ]


    y = 30

    for text in texts:

        cv2.putText(
            info,
            text,
            (10, y),

            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,

            (255, 255, 255),
            1,

            cv2.LINE_AA
        )

        y += 35


    final = np.vstack(
        [info, sheet]
    )


    output = (
        OUTPUT_DIR
        / f"{video_name.replace('.mp4', '')}_review.jpg"
    )


    cv2.imwrite(
        str(output),
        final
    )


# ============================================================
# CRÉER LES 30 PLANCHES
# ============================================================

for _, row in df.iterrows():

    create_sheet(row)


print("\nTerminé.")

print("Dossier :")
print(OUTPUT_DIR)