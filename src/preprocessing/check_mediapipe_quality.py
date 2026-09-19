from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = ROOT / "data" / "asl" / "asl_clean_technical.csv"

VIDEOS_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "videos"
)

OUTPUT_CSV = (
    ROOT
    / "data"
    / "asl"
    / "mediapipe_quality_report.csv"
)


# ============================================================
# CHARGER LE DATASET
# ============================================================

df = pd.read_csv(
    INPUT_CSV,
    dtype={"video": str}
)

# ============================================================
# TEST TEMPORAIRE SUR UNE SEULE VIDÉO
# IMPORTANT :
# on supprimera cette ligne après validation.
# ============================================================


print("Nombre de vidéos à analyser :", len(df))


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


# ============================================================
# PLUS LONGUE SÉQUENCE DE TRUE
# ============================================================

def longest_true_run(values):

    longest = 0
    current = 0

    for value in values:

        if value:
            current += 1
            longest = max(longest, current)

        else:
            current = 0

    return longest


# ============================================================
# MOYENNE SÉCURISÉE
# ============================================================

def safe_mean(values):

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


# ============================================================
# ANALYSER UNE VIDÉO
# ============================================================

def analyse_video(video_name, holistic):

    path = VIDEOS_DIR / video_name

    cap = cv2.VideoCapture(str(path))

    # Réinitialiser le tracking MediaPipe
    # entre deux vidéos différentes.
    holistic.reset()

    total_frames = 0

    pose_presence = []

    left_hand_presence = []
    right_hand_presence = []

    any_hand_presence = []
    both_hands_presence = []

    left_wrist_visibility = []
    right_wrist_visibility = []

    left_elbow_visibility = []
    right_elbow_visibility = []

    left_shoulder_visibility = []
    right_shoulder_visibility = []


    while True:

        ok, frame = cap.read()

        if not ok or frame is None:
            break

        total_frames += 1


        # OpenCV utilise BGR.
        # MediaPipe attend RGB.
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        results = holistic.process(rgb)

        rgb.flags.writeable = True


        # ====================================================
        # POSE
        # ====================================================

        pose_detected = (
            results.pose_landmarks is not None
        )

        pose_presence.append(
            pose_detected
        )


        # ====================================================
        # MAINS
        # ====================================================

        left_detected = (
            results.left_hand_landmarks is not None
        )

        right_detected = (
            results.right_hand_landmarks is not None
        )


        left_hand_presence.append(
            left_detected
        )

        right_hand_presence.append(
            right_detected
        )

        any_hand_presence.append(
            left_detected or right_detected
        )

        both_hands_presence.append(
            left_detected and right_detected
        )


        # ====================================================
        # VISIBILITÉ DES ARTICULATIONS DE LA POSE
        # ====================================================

        if pose_detected:

            landmarks = (
                results.pose_landmarks.landmark
            )


            left_wrist_visibility.append(
                landmarks[
                    mp_pose.PoseLandmark.LEFT_WRIST.value
                ].visibility
            )

            right_wrist_visibility.append(
                landmarks[
                    mp_pose.PoseLandmark.RIGHT_WRIST.value
                ].visibility
            )


            left_elbow_visibility.append(
                landmarks[
                    mp_pose.PoseLandmark.LEFT_ELBOW.value
                ].visibility
            )

            right_elbow_visibility.append(
                landmarks[
                    mp_pose.PoseLandmark.RIGHT_ELBOW.value
                ].visibility
            )


            left_shoulder_visibility.append(
                landmarks[
                    mp_pose.PoseLandmark.LEFT_SHOULDER.value
                ].visibility
            )

            right_shoulder_visibility.append(
                landmarks[
                    mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                ].visibility
            )


    cap.release()


    # ========================================================
    # CAS ANORMAL : AUCUNE FRAME
    # ========================================================

    if total_frames == 0:

        return {
            "video": video_name,
            "total_frames": 0,

            "pose_ratio": 0.0,

            "left_hand_ratio": 0.0,
            "right_hand_ratio": 0.0,

            "any_hand_ratio": 0.0,
            "both_hands_ratio": 0.0,

            "longest_hand_run_frames": 0,
            "longest_hand_run_ratio": 0.0,

            "longest_no_hand_run_frames": 0,
            "longest_no_hand_run_ratio": 0.0,

            "detection_changes": 0,
            "detection_change_ratio": 0.0,

            "middle_hand_ratio": 0.0,

            "left_wrist_visibility": np.nan,
            "right_wrist_visibility": np.nan,

            "left_elbow_visibility": np.nan,
            "right_elbow_visibility": np.nan,

            "left_shoulder_visibility": np.nan,
            "right_shoulder_visibility": np.nan
        }


    # ========================================================
    # RATIOS POSE / MAINS
    # ========================================================

    pose_ratio = (
        sum(pose_presence)
        / total_frames
    )

    left_hand_ratio = (
        sum(left_hand_presence)
        / total_frames
    )

    right_hand_ratio = (
        sum(right_hand_presence)
        / total_frames
    )

    any_hand_ratio = (
        sum(any_hand_presence)
        / total_frames
    )

    both_hands_ratio = (
        sum(both_hands_presence)
        / total_frames
    )


    # ========================================================
    # CONTINUITÉ DE DÉTECTION
    # ========================================================

    longest_hand_run = longest_true_run(
        any_hand_presence
    )

    no_hand_presence = [
        not value
        for value in any_hand_presence
    ]

    longest_no_hand_run = longest_true_run(
        no_hand_presence
    )


    longest_hand_run_ratio = (
        longest_hand_run
        / total_frames
    )

    longest_no_hand_run_ratio = (
        longest_no_hand_run
        / total_frames
    )


    # ========================================================
    # CHANGEMENTS DÉTECTÉ / NON DÉTECTÉ
    # ========================================================

    detection_changes = 0

    for i in range(
        1,
        len(any_hand_presence)
    ):

        if (
            any_hand_presence[i]
            != any_hand_presence[i - 1]
        ):

            detection_changes += 1


    detection_change_ratio = (
        detection_changes
        / max(total_frames - 1, 1)
    )


    # ========================================================
    # DÉTECTION DES MAINS DANS LE CENTRE DE LA VIDÉO
    #
    # On prend la partie située entre 20 % et 80 %.
    # ========================================================

    start = int(
        total_frames * 0.20
    )

    end = int(
        total_frames * 0.80
    )

    middle = any_hand_presence[
        start:end
    ]


    if len(middle) > 0:

        middle_hand_ratio = (
            sum(middle)
            / len(middle)
        )

    else:

        middle_hand_ratio = any_hand_ratio


    # ========================================================
    # RÉSULTAT
    # ========================================================

    return {

        "video":
            video_name,

        "total_frames":
            total_frames,


        # Pose

        "pose_ratio":
            pose_ratio,


        # Mains

        "left_hand_ratio":
            left_hand_ratio,

        "right_hand_ratio":
            right_hand_ratio,

        "any_hand_ratio":
            any_hand_ratio,

        "both_hands_ratio":
            both_hands_ratio,


        # Continuité

        "longest_hand_run_frames":
            longest_hand_run,

        "longest_hand_run_ratio":
            longest_hand_run_ratio,

        "longest_no_hand_run_frames":
            longest_no_hand_run,

        "longest_no_hand_run_ratio":
            longest_no_hand_run_ratio,


        # Stabilité de détection

        "detection_changes":
            detection_changes,

        "detection_change_ratio":
            detection_change_ratio,


        # Partie centrale

        "middle_hand_ratio":
            middle_hand_ratio,


        # Visibilité des articulations

        "left_wrist_visibility":
            safe_mean(left_wrist_visibility),

        "right_wrist_visibility":
            safe_mean(right_wrist_visibility),

        "left_elbow_visibility":
            safe_mean(left_elbow_visibility),

        "right_elbow_visibility":
            safe_mean(right_elbow_visibility),

        "left_shoulder_visibility":
            safe_mean(left_shoulder_visibility),

        "right_shoulder_visibility":
            safe_mean(right_shoulder_visibility)
    }


# ============================================================
# LANCER MEDIAPIPE
# ============================================================

results = []


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

        df.iterrows(),

        total=len(df),

        desc="Analyse MediaPipe"

    ):

        result = analyse_video(
            row["video"],
            holistic
        )

        result["gloss"] = row["gloss"]
        result["split"] = row["split"]

        results.append(result)


# ============================================================
# CRÉER LE RAPPORT
# ============================================================

report = pd.DataFrame(results)


first_columns = [
    "video",
    "gloss",
    "split",
    "total_frames"
]


remaining_columns = [
    column
    for column in report.columns
    if column not in first_columns
]


report = report[
    first_columns
    + remaining_columns
]


numeric_columns = report.select_dtypes(
    include="number"
).columns


report[numeric_columns] = (
    report[numeric_columns]
    .round(4)
)


report.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# AFFICHAGE DES RÉSULTATS
# ============================================================

print("\n")
print("=" * 60)
print("RÉSULTAT MEDIAPIPE")
print("=" * 60)

print(
    "Vidéos analysées :",
    len(report)
)


important_columns = [

    "pose_ratio",

    "any_hand_ratio",

    "both_hands_ratio",

    "middle_hand_ratio",

    "longest_hand_run_ratio",

    "longest_no_hand_run_ratio",

    "detection_change_ratio",

    "left_wrist_visibility",

    "right_wrist_visibility"
]


print(
    "\n========== STATISTIQUES =========="
)

print(
    report[
        important_columns
    ].describe()
)


print(
    "\n========== AUCUNE MAIN DÉTECTÉE =========="
)

print(
    int(
        (
            report["any_hand_ratio"] == 0
        ).sum()
    )
)


print(
    "\n========== AU MOINS UNE MAIN < 25 % =========="
)

print(
    int(
        (
            report["any_hand_ratio"] < 0.25
        ).sum()
    )
)


print(
    "\n========== AU MOINS UNE MAIN < 50 % =========="
)

print(
    int(
        (
            report["any_hand_ratio"] < 0.50
        ).sum()
    )
)


print(
    "\n========== POSE < 50 % =========="
)

print(
    int(
        (
            report["pose_ratio"] < 0.50
        ).sum()
    )
)


print("\nRapport créé :")
print(OUTPUT_CSV)
