from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

COMPARISON_CSV = (
    ROOT
    / "data"
    / "asl"
    / "northtexas_mediapipe_comparison.csv"
)

OLD_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "videos"
)

NEW_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "videos_northtexas_recut"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "northtexas_old_vs_recut"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


N_FRAMES = 10
WIDTH = 300
HEIGHT = 170


# ============================================================
# EXTRAIRE 10 FRAMES
# ============================================================

def extract_frames(path):

    cap = cv2.VideoCapture(str(path))

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total <= 0:
        cap.release()
        return []

    positions = np.linspace(
        0,
        total - 1,
        N_FRAMES,
        dtype=int
    )

    frames = []

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
            (WIDTH, HEIGHT)
        )

        cv2.putText(
            frame,
            f"{pos}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        frames.append(frame)

    cap.release()

    while len(frames) < N_FRAMES:

        frames.append(
            np.zeros(
                (HEIGHT, WIDTH, 3),
                dtype=np.uint8
            )
        )

    return frames


# ============================================================
# CHARGER LES 10 PIRES DELTAS
# ============================================================

df = pd.read_csv(COMPARISON_CSV)

df = df.sort_values(
    "delta_any_hand",
    ascending=True
).head(10)


# ============================================================
# CRÉER LES COMPARAISONS
# ============================================================

for _, row in df.iterrows():

    video = row["video"]
    gloss = row["gloss"]

    old_frames = extract_frames(
        OLD_DIR / video
    )

    new_frames = extract_frames(
        NEW_DIR / video
    )

    if not old_frames or not new_frames:
        print("Impossible :", video)
        continue


    old_sheet = np.vstack([
        np.hstack(old_frames[0:5]),
        np.hstack(old_frames[5:10])
    ])

    new_sheet = np.vstack([
        np.hstack(new_frames[0:5]),
        np.hstack(new_frames[5:10])
    ])


    label_height = 55

    old_label = np.zeros(
        (
            label_height,
            old_sheet.shape[1],
            3
        ),
        dtype=np.uint8
    )

    new_label = np.zeros(
        (
            label_height,
            new_sheet.shape[1],
            3
        ),
        dtype=np.uint8
    )


    cv2.putText(
        old_label,
        (
            f"ANCIEN | {video} | {gloss} | "
            f"hand={row['old_any_hand_ratio']:.3f}"
        ),
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        new_label,
        (
            f"RECUT | {video} | {gloss} | "
            f"hand={row['new_any_hand_ratio']:.3f}"
        ),
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    final = np.vstack([
        old_label,
        old_sheet,
        new_label,
        new_sheet
    ])


    output = (
        OUTPUT_DIR
        / video.replace(
            ".mp4",
            "_comparison.jpg"
        )
    )

    cv2.imwrite(
        str(output),
        final
    )

    print(
        video,
        gloss,
        "delta =",
        round(
            row["delta_any_hand"],
            4
        )
    )


print("\nCréé dans :")
print(OUTPUT_DIR)