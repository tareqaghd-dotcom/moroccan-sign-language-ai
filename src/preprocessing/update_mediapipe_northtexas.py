from pathlib import Path
import json
import shutil

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = (
    ROOT
    / "data"
    / "asl"
    / "WLASL100.json"
)

TECHNICAL_CSV = (
    ROOT
    / "data"
    / "asl"
    / "asl_clean_technical.csv"
)

MEDIAPIPE_CSV = (
    ROOT
    / "data"
    / "asl"
    / "mediapipe_quality_report.csv"
)

BACKUP_CSV = (
    ROOT
    / "data"
    / "asl"
    / "mediapipe_quality_report_before_northtexas_fix.csv"
)

VIDEOS_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "videos"
)


# ============================================================
# 1. TROUVER LES 51 VIDÉOS NORTHTEXAS
# ============================================================

with open(
    JSON_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


northtexas_ids = set()


for item in data:

    for inst in item["instances"]:

        if inst.get("source") == "northtexas":

            video_id = str(
                inst["video_id"]
            ).zfill(5)

            northtexas_ids.add(
                f"{video_id}.mp4"
            )


print(
    "Vidéos northtexas trouvées :",
    len(northtexas_ids)
)


if len(northtexas_ids) != 51:

    raise SystemExit(
        f"ERREUR : {len(northtexas_ids)} vidéos au lieu de 51"
    )


# ============================================================
# 2. CHARGER LE DATASET TECHNIQUEMENT VALIDE
# ============================================================

technical = pd.read_csv(
    TECHNICAL_CSV,
    dtype={"video": str}
)


targets = technical[
    technical["video"].isin(
        northtexas_ids
    )
].copy()


print(
    "Vidéos northtexas techniquement valides :",
    len(targets)
)


if len(targets) != 51:

    raise SystemExit(
        f"ERREUR : seulement {len(targets)} vidéos northtexas valides"
    )


# ============================================================
# 3. SAUVEGARDER L'ANCIEN RAPPORT MEDIAPIPE
# ============================================================

if not BACKUP_CSV.exists():

    shutil.copy2(
        MEDIAPIPE_CSV,
        BACKUP_CSV
    )

    print(
        "Backup MediaPipe créé :",
        BACKUP_CSV
    )

else:

    print(
        "Backup MediaPipe déjà présent."
    )


old_report = pd.read_csv(
    MEDIAPIPE_CSV
)


# ============================================================
# FONCTIONS
# ============================================================

def longest_true_run(values):

    longest = 0
    current = 0

    for value in values:

        if value:
            current += 1
            longest = max(
                longest,
                current
            )

        else:
            current = 0

    return longest


def safe_mean(values):

    if not values:
        return np.nan

    return float(
        np.mean(values)
    )


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose


def analyse_video(
    video_name,
    holistic
):

    path = (
        VIDEOS_DIR
        / video_name
    )

    cap = cv2.VideoCapture(
        str(path)
    )

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


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        rgb.flags.writeable = False

        results = holistic.process(
            rgb
        )

        rgb.flags.writeable = True


        # ----------------------------------------------------
        # POSE
        # ----------------------------------------------------

        pose_detected = (
            results.pose_landmarks
            is not None
        )

        pose_presence.append(
            pose_detected
        )


        # ----------------------------------------------------
        # MAINS
        # ----------------------------------------------------

        left_detected = (
            results.left_hand_landmarks
            is not None
        )

        right_detected = (
            results.right_hand_landmarks
            is not None
        )


        left_hand_presence.append(
            left_detected
        )

        right_hand_presence.append(
            right_detected
        )


        any_hand_presence.append(
            left_detected
            or right_detected
        )

        both_hands_presence.append(
            left_detected
            and right_detected
        )


        # ----------------------------------------------------
        # VISIBILITÉ POSE
        # ----------------------------------------------------

        if pose_detected:

            landmarks = (
                results
                .pose_landmarks
                .landmark
            )


            left_wrist_visibility.append(
                landmarks[
                    mp_pose
                    .PoseLandmark
                    .LEFT_WRIST
                    .value
                ].visibility
            )


            right_wrist_visibility.append(
                landmarks[
                    mp_pose
                    .PoseLandmark
                    .RIGHT_WRIST
                    .value
                ].visibility
            )


            left_elbow_visibility.append(
                landmarks[
                    mp_pose
                    .PoseLandmark
                    .LEFT_ELBOW
                    .value
                ].visibility
            )


            right_elbow_visibility.append(
                landmarks[
                    mp_pose
                    .PoseLandmark
                    .RIGHT_ELBOW
                    .value
                ].visibility
            )


            left_shoulder_visibility.append(
                landmarks[
                    mp_pose
                    .PoseLandmark
                    .LEFT_SHOULDER
                    .value
                ].visibility
            )


            right_shoulder_visibility.append(
                landmarks[
                    mp_pose
                    .PoseLandmark
                    .RIGHT_SHOULDER
                    .value
                ].visibility
            )


    cap.release()


    if total_frames == 0:

        raise RuntimeError(
            f"Aucune frame : {video_name}"
        )


    # ========================================================
    # RATIOS
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
    # CONTINUITÉ
    # ========================================================

    longest_hand_run = (
        longest_true_run(
            any_hand_presence
        )
    )


    no_hand_presence = [
        not x
        for x in any_hand_presence
    ]


    longest_no_hand_run = (
        longest_true_run(
            no_hand_presence
        )
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
    # CHANGEMENTS DÉTECTION / NON-DÉTECTION
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
        / max(
            total_frames - 1,
            1
        )
    )


    # ========================================================
    # PARTIE CENTRALE 20 % -> 80 %
    # ========================================================

    start = int(
        total_frames * 0.20
    )

    end = int(
        total_frames * 0.80
    )


    middle = (
        any_hand_presence[
            start:end
        ]
    )


    if len(middle) > 0:

        middle_hand_ratio = (
            sum(middle)
            / len(middle)
        )

    else:

        middle_hand_ratio = (
            any_hand_ratio
        )


    # ========================================================
    # RÉSULTAT
    # ========================================================

    return {

        "video":
            video_name,

        "total_frames":
            total_frames,

        "pose_ratio":
            pose_ratio,

        "left_hand_ratio":
            left_hand_ratio,

        "right_hand_ratio":
            right_hand_ratio,

        "any_hand_ratio":
            any_hand_ratio,

        "both_hands_ratio":
            both_hands_ratio,

        "longest_hand_run_frames":
            longest_hand_run,

        "longest_hand_run_ratio":
            longest_hand_run_ratio,

        "longest_no_hand_run_frames":
            longest_no_hand_run,

        "longest_no_hand_run_ratio":
            longest_no_hand_run_ratio,

        "detection_changes":
            detection_changes,

        "detection_change_ratio":
            detection_change_ratio,

        "middle_hand_ratio":
            middle_hand_ratio,

        "left_wrist_visibility":
            safe_mean(
                left_wrist_visibility
            ),

        "right_wrist_visibility":
            safe_mean(
                right_wrist_visibility
            ),

        "left_elbow_visibility":
            safe_mean(
                left_elbow_visibility
            ),

        "right_elbow_visibility":
            safe_mean(
                right_elbow_visibility
            ),

        "left_shoulder_visibility":
            safe_mean(
                left_shoulder_visibility
            ),

        "right_shoulder_visibility":
            safe_mean(
                right_shoulder_visibility
            )
    }


# ============================================================
# 4. ANALYSER SEULEMENT LES 51
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

        targets.iterrows(),

        total=len(targets),

        desc="MediaPipe northtexas corrigé"

    ):

        result = analyse_video(
            row["video"],
            holistic
        )

        result["gloss"] = (
            row["gloss"]
        )

        result["split"] = (
            row["split"]
        )

        results.append(
            result
        )


new_report = pd.DataFrame(
    results
)


# ============================================================
# 5. MÊME ORDRE DE COLONNES QUE LE RAPPORT ORIGINAL
# ============================================================

missing = [
    col
    for col in old_report.columns
    if col not in new_report.columns
]


if missing:

    raise SystemExit(
        "Colonnes manquantes : "
        + str(missing)
    )


new_report = new_report[
    old_report.columns
]


# ============================================================
# 6. COMPARAISON AVANT / APRÈS
# ============================================================

old_northtexas = old_report[
    old_report["video"].isin(
        northtexas_ids
    )
].copy()


print("\n")
print("=" * 70)
print("MEDIAPIPE NORTHTEXAS APRÈS CORRECTION")
print("=" * 70)


print(
    "Ancienne moyenne any_hand :",
    round(
        old_northtexas[
            "any_hand_ratio"
        ].mean(),
        4
    )
)


print(
    "Nouvelle moyenne any_hand :",
    round(
        new_report[
            "any_hand_ratio"
        ].mean(),
        4
    )
)


print(
    "\nAncienne moyenne pose :",
    round(
        old_northtexas[
            "pose_ratio"
        ].mean(),
        4
    )
)


print(
    "Nouvelle moyenne pose :",
    round(
        new_report[
            "pose_ratio"
        ].mean(),
        4
    )
)


print(
    "\nAnciennes vidéos sans main :",
    int(
        (
            old_northtexas[
                "any_hand_ratio"
            ] == 0
        ).sum()
    )
)


print(
    "Nouvelles vidéos sans main :",
    int(
        (
            new_report[
                "any_hand_ratio"
            ] == 0
        ).sum()
    )
)


print(
    "\nNouvelles vidéos main < 25 % :",
    int(
        (
            new_report[
                "any_hand_ratio"
            ] < 0.25
        ).sum()
    )
)


# ============================================================
# 7. REMPLACER UNIQUEMENT LES 51 LIGNES
# ============================================================

remaining = old_report[
    ~old_report["video"].isin(
        northtexas_ids
    )
].copy()


updated = pd.concat(
    [
        remaining,
        new_report
    ],
    ignore_index=True
)


if len(updated) != 1023:

    raise SystemExit(
        f"ERREUR : rapport final = {len(updated)} lignes"
    )


# garder un ordre stable
updated = updated.sort_values(
    ["gloss", "video"]
).reset_index(
    drop=True
)


numeric_columns = (
    updated
    .select_dtypes(
        include="number"
    )
    .columns
)


updated[
    numeric_columns
] = (
    updated[
        numeric_columns
    ]
    .round(4)
)


updated.to_csv(
    MEDIAPIPE_CSV,
    index=False
)


print(
    "\nRapport MediaPipe final :",
    len(updated),
    "vidéos"
)

print(
    "Sauvegardé dans :",
    MEDIAPIPE_CSV
)

print(
    "\nBackup ancien rapport :",
    BACKUP_CSV
)