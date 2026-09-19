from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "technical_clean.csv"
)

VIDEOS_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "candidate_batch1_processed"
)

OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "mediapipe_quality_report.csv"
)


# ============================================================
# DONNÉES
# ============================================================

df = pd.read_csv(
    INPUT,
    dtype={"video": str}
)


print("=" * 80)
print("MEDIAPIPE — CANDIDATE BATCH 1")
print("=" * 80)

print(
    "Vidéos à analyser :",
    len(df)
)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = (
    mp.solutions.holistic
)

holistic = mp_holistic.Holistic(

    static_image_mode=False,

    model_complexity=1,

    smooth_landmarks=True,

    enable_segmentation=False,

    refine_face_landmarks=False,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


rows = []


# ============================================================
# FONCTION VISIBILITE POSE
# ============================================================

def get_pose_visibility(
    pose_landmarks,
    landmark_index
):

    if pose_landmarks is None:
        return np.nan

    try:

        return float(
            pose_landmarks
            .landmark[
                landmark_index
            ]
            .visibility
        )

    except Exception:

        return np.nan


# ============================================================
# BOUCLE VIDEOS
# ============================================================

for position, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    video = row["video"]

    gloss = row["gloss"]

    path = (
        VIDEOS_DIR
        / video
    )


    print(
        f"[{position}/{len(df)}]",
        video,
        gloss
    )


    cap = cv2.VideoCapture(
        str(path)
    )


    if not cap.isOpened():

        rows.append({

            "video":
                video,

            "gloss":
                gloss,

            "frames":
                0,

            "pose_ratio":
                0,

            "left_hand_ratio":
                0,

            "right_hand_ratio":
                0,

            "any_hand_ratio":
                0,

            "both_hands_ratio":
                0,

            "middle_hand_ratio":
                0,

            "longest_no_hand_run":
                0,

            "longest_no_hand_run_ratio":
                1,

            "hand_detection_changes":
                0,

            "left_wrist_visibility":
                np.nan,

            "right_wrist_visibility":
                np.nan,

            "left_elbow_visibility":
                np.nan,

            "right_elbow_visibility":
                np.nan,

            "left_shoulder_visibility":
                np.nan,

            "right_shoulder_visibility":
                np.nan,

            "status":
                "OPEN_FAILED"
        })

        continue


    # Reset tracking entre les vidéos
    try:

        holistic.reset()

    except Exception:
        pass


    frames = 0

    pose_frames = 0

    left_hand_frames = 0

    right_hand_frames = 0

    any_hand_frames = 0

    both_hand_frames = 0


    hand_presence = []


    left_wrist_vis = []
    right_wrist_vis = []

    left_elbow_vis = []
    right_elbow_vis = []

    left_shoulder_vis = []
    right_shoulder_vis = []


    # ========================================================
    # FRAME PAR FRAME
    # ========================================================

    while True:

        ok, frame = cap.read()

        if not ok:
            break


        frames += 1


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        results = holistic.process(
            rgb
        )


        pose_present = (
            results.pose_landmarks
            is not None
        )


        left_present = (
            results.left_hand_landmarks
            is not None
        )


        right_present = (
            results.right_hand_landmarks
            is not None
        )


        any_present = (
            left_present
            or right_present
        )


        both_present = (
            left_present
            and right_present
        )


        if pose_present:
            pose_frames += 1

        if left_present:
            left_hand_frames += 1

        if right_present:
            right_hand_frames += 1

        if any_present:
            any_hand_frames += 1

        if both_present:
            both_hand_frames += 1


        hand_presence.append(
            1
            if any_present
            else 0
        )


        # ====================================================
        # VISIBILITE DES ARTICULATIONS
        # ====================================================

        if pose_present:

            pose = (
                results.pose_landmarks
            )


            # indices MediaPipe Pose :
            # 11 = left shoulder
            # 12 = right shoulder
            # 13 = left elbow
            # 14 = right elbow
            # 15 = left wrist
            # 16 = right wrist

            left_shoulder_vis.append(
                get_pose_visibility(
                    pose,
                    11
                )
            )

            right_shoulder_vis.append(
                get_pose_visibility(
                    pose,
                    12
                )
            )

            left_elbow_vis.append(
                get_pose_visibility(
                    pose,
                    13
                )
            )

            right_elbow_vis.append(
                get_pose_visibility(
                    pose,
                    14
                )
            )

            left_wrist_vis.append(
                get_pose_visibility(
                    pose,
                    15
                )
            )

            right_wrist_vis.append(
                get_pose_visibility(
                    pose,
                    16
                )
            )


    cap.release()


    # ========================================================
    # RATIOS
    # ========================================================

    if frames > 0:

        pose_ratio = (
            pose_frames
            / frames
        )

        left_ratio = (
            left_hand_frames
            / frames
        )

        right_ratio = (
            right_hand_frames
            / frames
        )

        any_ratio = (
            any_hand_frames
            / frames
        )

        both_ratio = (
            both_hand_frames
            / frames
        )

    else:

        pose_ratio = 0
        left_ratio = 0
        right_ratio = 0
        any_ratio = 0
        both_ratio = 0


    # ========================================================
    # ZONE CENTRALE 20% → 80%
    # ========================================================

    if frames > 0:

        start_middle = int(
            frames * 0.20
        )

        end_middle = int(
            frames * 0.80
        )


        middle = hand_presence[
            start_middle:
            end_middle
        ]


        if len(middle) > 0:

            middle_hand_ratio = (
                sum(middle)
                / len(middle)
            )

        else:

            middle_hand_ratio = 0

    else:

        middle_hand_ratio = 0


    # ========================================================
    # PLUS LONGUE PERIODE SANS MAIN
    # ========================================================

    longest_no_hand_run = 0

    current_no_hand_run = 0


    for detected in hand_presence:

        if detected == 0:

            current_no_hand_run += 1

            longest_no_hand_run = max(
                longest_no_hand_run,
                current_no_hand_run
            )

        else:

            current_no_hand_run = 0


    if frames > 0:

        longest_no_hand_ratio = (
            longest_no_hand_run
            / frames
        )

    else:

        longest_no_hand_ratio = 1


    # ========================================================
    # CHANGEMENTS DETECTION / PAS DETECTION
    # ========================================================

    detection_changes = 0


    for i in range(
        1,
        len(hand_presence)
    ):

        if (
            hand_presence[i]
            != hand_presence[i - 1]
        ):

            detection_changes += 1


    # ========================================================
    # MOYENNE VISIBILITE
    # ========================================================

    def mean_or_nan(values):

        if not values:
            return np.nan

        return float(
            np.nanmean(
                values
            )
        )


    rows.append({

        "video":
            video,

        "gloss":
            gloss,

        "frames":
            frames,

        "pose_ratio":
            pose_ratio,

        "left_hand_ratio":
            left_ratio,

        "right_hand_ratio":
            right_ratio,

        "any_hand_ratio":
            any_ratio,

        "both_hands_ratio":
            both_ratio,

        "middle_hand_ratio":
            middle_hand_ratio,

        "longest_no_hand_run":
            longest_no_hand_run,

        "longest_no_hand_run_ratio":
            longest_no_hand_ratio,

        "hand_detection_changes":
            detection_changes,

        "left_wrist_visibility":
            mean_or_nan(
                left_wrist_vis
            ),

        "right_wrist_visibility":
            mean_or_nan(
                right_wrist_vis
            ),

        "left_elbow_visibility":
            mean_or_nan(
                left_elbow_vis
            ),

        "right_elbow_visibility":
            mean_or_nan(
                right_elbow_vis
            ),

        "left_shoulder_visibility":
            mean_or_nan(
                left_shoulder_vis
            ),

        "right_shoulder_visibility":
            mean_or_nan(
                right_shoulder_vis
            ),

        "status":
            "OK"
    })


# ============================================================
# FIN MEDIAPIPE
# ============================================================

holistic.close()


# ============================================================
# RAPPORT
# ============================================================

report = pd.DataFrame(
    rows
)


report.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# STATISTIQUES
# ============================================================

print("\n")
print("=" * 80)
print("RÉSULTAT MEDIAPIPE")
print("=" * 80)


print(
    "Vidéos analysées :",
    len(report)
)


print(
    "Pose absente sur toute la vidéo :",
    int(
        (
            report[
                "pose_ratio"
            ]
            == 0
        ).sum()
    )
)


print(
    "Aucune main détectée :",
    int(
        (
            report[
                "any_hand_ratio"
            ]
            == 0
        ).sum()
    )
)


print(
    "Main détectée < 25 % :",
    int(
        (
            report[
                "any_hand_ratio"
            ]
            < 0.25
        ).sum()
    )
)


print(
    "Main détectée < 50 % :",
    int(
        (
            report[
                "any_hand_ratio"
            ]
            < 0.50
        ).sum()
    )
)


print(
    "Pose détectée < 50 % :",
    int(
        (
            report[
                "pose_ratio"
            ]
            < 0.50
        ).sum()
    )
)


print(
    "Aucune main dans la zone centrale :",
    int(
        (
            report[
                "middle_hand_ratio"
            ]
            == 0
        ).sum()
    )
)


print(
    "\n========== DISTRIBUTION =========="
)


print(
    "\nPose ratio :"
)

print(
    report[
        "pose_ratio"
    ]
    .describe()
)


print(
    "\nAny hand ratio :"
)

print(
    report[
        "any_hand_ratio"
    ]
    .describe()
)


print(
    "\nBoth hands ratio :"
)

print(
    report[
        "both_hands_ratio"
    ]
    .describe()
)


print(
    "\nMiddle hand ratio :"
)

print(
    report[
        "middle_hand_ratio"
    ]
    .describe()
)


# ============================================================
# LES CAS LES PLUS FAIBLES
# ============================================================

print(
    "\n========== 30 VIDEOS AVEC LE MOINS DE DETECTION DE MAINS =========="
)

print(
    report[
        [
            "video",
            "gloss",
            "pose_ratio",
            "any_hand_ratio",
            "middle_hand_ratio",
            "longest_no_hand_run_ratio"
        ]
    ]
    .sort_values(
        "any_hand_ratio"
    )
    .head(30)
    .to_string(
        index=False
    )
)


print(
    "\nRapport créé :"
)

print(
    OUTPUT
)
