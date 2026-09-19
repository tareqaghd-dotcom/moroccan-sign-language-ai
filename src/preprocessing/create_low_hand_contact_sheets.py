from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CSV = (
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
    / "low_hand_review"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(CSV)

df = (
    df[
        df["any_hand_ratio"] < 0.25
    ]
    .sort_values(
        "any_hand_ratio"
    )
)


N = 12
WIDTH = 320
HEIGHT = 180


for _, row in df.iterrows():

    video = row["video"]
    gloss = row["gloss"]

    path = VIDEOS_DIR / video

    cap = cv2.VideoCapture(
        str(path)
    )

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    positions = np.linspace(
        0,
        max(total - 1, 0),
        N,
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
            f"frame {pos}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        frames.append(frame)


    cap.release()


    if not frames:
        continue


    while len(frames) < N:

        frames.append(
            np.zeros(
                (HEIGHT, WIDTH, 3),
                dtype=np.uint8
            )
        )


    row1 = np.hstack(
        frames[:6]
    )

    row2 = np.hstack(
        frames[6:12]
    )


    header = np.zeros(
        (
            70,
            row1.shape[1],
            3
        ),
        dtype=np.uint8
    )


    text = (
        f"{video} | {gloss} | "
        f"hand={row['any_hand_ratio']:.3f} | "
        f"middle={row['middle_hand_ratio']:.3f}"
    )


    cv2.putText(
        header,
        text,
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    sheet = np.vstack(
        [
            header,
            row1,
            row2
        ]
    )


    output = (
        OUTPUT_DIR
        / video.replace(
            ".mp4",
            "_review.jpg"
        )
    )


    cv2.imwrite(
        str(output),
        sheet
    )


    print(
        video,
        gloss,
        "| any_hand =",
        round(
            row["any_hand_ratio"],
            4
        )
    )


print("\nCréé dans :")
print(OUTPUT_DIR)