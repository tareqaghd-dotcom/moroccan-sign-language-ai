from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

OLD_REPORT = (
    ROOT
    / "data"
    / "asl"
    / "mediapipe_quality_report.csv"
)

RECUT_REPORT = (
    ROOT
    / "data"
    / "asl"
    / "northtexas_recut_report.csv"
)

RECUT_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "videos_northtexas_recut"
)

OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "northtexas_mediapipe_comparison.csv"
)


# ============================================================
# 1. CHARGER LES DONNÉES
# ============================================================

old = pd.read_csv(OLD_REPORT)

recut_info = pd.read_csv(RECUT_REPORT)


northtexas_videos = set(
    recut_info["video"]
)


old = old[
    old["video"].isin(northtexas_videos)
].copy()


# ============================================================
# 2. MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic


def analyse_video(path, holistic):

    cap = cv2.VideoCapture(str(path))

    holistic.reset()

    total = 0
    pose = 0

    left_hand = 0
    right_hand = 0

    any_hand = 0
    both_hands = 0

    any_hand_presence = []


    while True:

        ok, frame = cap.read()

        if not ok or frame is None:
            break

        total += 1

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = holistic.process(rgb)


        pose_detected = (
            results.pose_landmarks is not None
        )

        left_detected = (
            results.left_hand_landmarks is not None
        )

        right_detected = (
            results.right_hand_landmarks is not None
        )


        if pose_detected:
            pose += 1

        if left_detected:
            left_hand += 1

        if right_detected:
            right_hand += 1

        if left_detected or right_detected:
            any_hand += 1

        if left_detected and right_detected:
            both_hands += 1


        any_hand_presence.append(
            left_detected or right_detected
        )


    cap.release()


    if total == 0:

        return {
            "new_total_frames": 0,
            "new_pose_ratio": 0.0,
            "new_left_hand_ratio": 0.0,
            "new_right_hand_ratio": 0.0,
            "new_any_hand_ratio": 0.0,
            "new_both_hands_ratio": 0.0
        }


    return {

        "new_total_frames":
            total,

        "new_pose_ratio":
            pose / total,

        "new_left_hand_ratio":
            left_hand / total,

        "new_right_hand_ratio":
            right_hand / total,

        "new_any_hand_ratio":
            any_hand / total,

        "new_both_hands_ratio":
            both_hands / total
    }


# ============================================================
# 3. ANALYSER LES 51 RECUTS
# ============================================================

rows = []


with mp_holistic.Holistic(

    static_image_mode=False,

    model_complexity=1,

    smooth_landmarks=True,

    enable_segmentation=False,

    refine_face_landmarks=False,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5

) as holistic:


    for _, row in tqdm(
        recut_info.iterrows(),
        total=len(recut_info),
        desc="MediaPipe recuts northtexas"
    ):

        video = row["video"]

        result = analyse_video(
            RECUT_DIR / video,
            holistic
        )

        result["video"] = video
        result["gloss"] = row["gloss"]

        rows.append(result)


new = pd.DataFrame(rows)


# ============================================================
# 4. ANCIENNES MESURES
# ============================================================

old = old[
    [
        "video",
        "pose_ratio",
        "left_hand_ratio",
        "right_hand_ratio",
        "any_hand_ratio",
        "both_hands_ratio"
    ]
].copy()


old = old.rename(
    columns={
        "pose_ratio":
            "old_pose_ratio",

        "left_hand_ratio":
            "old_left_hand_ratio",

        "right_hand_ratio":
            "old_right_hand_ratio",

        "any_hand_ratio":
            "old_any_hand_ratio",

        "both_hands_ratio":
            "old_both_hands_ratio"
    }
)


# ============================================================
# 5. COMPARAISON
# ============================================================

comparison = new.merge(
    old,
    on="video",
    how="left"
)


comparison["delta_any_hand"] = (
    comparison["new_any_hand_ratio"]
    - comparison["old_any_hand_ratio"]
)


comparison["delta_pose"] = (
    comparison["new_pose_ratio"]
    - comparison["old_pose_ratio"]
)


comparison = comparison.sort_values(
    "delta_any_hand",
    ascending=False
)


comparison.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# 6. STATISTIQUES
# ============================================================

print("\n")
print("=" * 70)
print("COMPARAISON MEDIAPIPE NORTHTEXAS")
print("=" * 70)

print(
    "Vidéos comparées :",
    len(comparison)
)


print("\n========== ANY HAND ==========")

print(
    "Ancienne moyenne :",
    round(
        comparison[
            "old_any_hand_ratio"
        ].mean(),
        4
    )
)

print(
    "Nouvelle moyenne :",
    round(
        comparison[
            "new_any_hand_ratio"
        ].mean(),
        4
    )
)

print(
    "Amélioration moyenne :",
    round(
        comparison[
            "delta_any_hand"
        ].mean(),
        4
    )
)


print(
    "\nVidéos améliorées :",
    int(
        (
            comparison["delta_any_hand"]
            > 0
        ).sum()
    )
)

print(
    "Vidéos identiques :",
    int(
        (
            comparison["delta_any_hand"]
            == 0
        ).sum()
    )
)

print(
    "Vidéos moins bonnes :",
    int(
        (
            comparison["delta_any_hand"]
            < 0
        ).sum()
    )
)


print(
    "\nAncien any_hand = 0 :",
    int(
        (
            comparison["old_any_hand_ratio"]
            == 0
        ).sum()
    )
)

print(
    "Nouveau any_hand = 0 :",
    int(
        (
            comparison["new_any_hand_ratio"]
            == 0
        ).sum()
    )
)


print("\n========== POSE ==========")

print(
    "Ancienne moyenne :",
    round(
        comparison[
            "old_pose_ratio"
        ].mean(),
        4
    )
)

print(
    "Nouvelle moyenne :",
    round(
        comparison[
            "new_pose_ratio"
        ].mean(),
        4
    )
)


print("\n========== 20 PLUS GRANDES AMÉLIORATIONS ==========")

print(
    comparison[
        [
            "video",
            "gloss",

            "old_any_hand_ratio",
            "new_any_hand_ratio",
            "delta_any_hand",

            "old_pose_ratio",
            "new_pose_ratio"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


print("\n========== 10 PIRES DELTAS ==========")

print(
    comparison[
        [
            "video",
            "gloss",

            "old_any_hand_ratio",
            "new_any_hand_ratio",
            "delta_any_hand"
        ]
    ]
    .sort_values(
        "delta_any_hand"
    )
    .head(10)
    .to_string(index=False)
)


print("\nFichier créé :")
print(OUTPUT)