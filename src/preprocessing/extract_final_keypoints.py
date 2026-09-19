from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    ROOT
    / "data"
    / "asl"
    / "final_wlasl100_split_v2.csv"
)

VIDEOS_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "final_wlasl100_videos"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "asl"
    / "keypoints_raw"
)

REPORT_FILE = (
    ROOT
    / "data"
    / "asl"
    / "keypoints_extraction_report.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic


# ============================================================
# EXTRACTION D'UNE FRAME
#
# Ordre :
#
# 0:99      pose
# 99:162    main gauche
# 162:225   main droite
#
# Chaque landmark = x, y, z
# ============================================================

def extract_keypoints(results):

    # --------------------------------------------------------
    # POSE : 33 × 3 = 99
    # --------------------------------------------------------

    if results.pose_landmarks:

        pose = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z
                ]
                for lm
                in results.pose_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        pose = np.zeros(
            33 * 3,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # MAIN GAUCHE : 21 × 3 = 63
    # --------------------------------------------------------

    if results.left_hand_landmarks:

        left = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z
                ]
                for lm
                in results.left_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        left = np.zeros(
            21 * 3,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # MAIN DROITE : 21 × 3 = 63
    # --------------------------------------------------------

    if results.right_hand_landmarks:

        right = np.array(
            [
                [
                    lm.x,
                    lm.y,
                    lm.z
                ]
                for lm
                in results.right_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        right = np.zeros(
            21 * 3,
            dtype=np.float32
        )


    keypoints = np.concatenate(
        [
            pose,
            left,
            right
        ]
    ).astype(
        np.float32
    )


    # Sécurité
    if keypoints.shape != (225,):

        raise RuntimeError(
            f"Dimension incorrecte : {keypoints.shape}"
        )


    return keypoints


# ============================================================
# CHARGEMENT CSV
# ============================================================

df = pd.read_csv(
    CSV_FILE,
    dtype={"video": str}
)


print("=" * 85)
print("EXTRACTION KEYPOINTS — WLASL100 FINAL")
print("=" * 85)

print(
    "Vidéos :",
    len(df)
)

print(
    "Classes :",
    df["gloss"].nunique()
)

print(
    "\nSplits :"
)

print(
    df["split"]
    .value_counts()
)


# ============================================================
# MODELE MEDIAPIPE
# ============================================================

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
# BOUCLE VIDEOS
# ============================================================

for position, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    video = str(
        row["video"]
    )

    gloss = str(
        row["gloss"]
    )

    split = str(
        row["split"]
    )


    video_path = (
        VIDEOS_DIR
        / video
    )


    # Dossier par split
    output_split_dir = (
        OUTPUT_DIR
        / split
    )

    output_split_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_file = (
        output_split_dir
        / f"{Path(video).stem}.npy"
    )


    print(
        f"[{position}/{len(df)}]",
        video,
        gloss,
        "|",
        split
    )


    result_row = {

        "video":
            video,

        "gloss":
            gloss,

        "split":
            split,

        "signer_id":
            row["signer_id"],

        "frames":
            0,

        "feature_dim":
            225,

        "pose_detected_frames":
            0,

        "left_hand_detected_frames":
            0,

        "right_hand_detected_frames":
            0,

        "any_hand_detected_frames":
            0,

        "success":
            False,

        "status":
            "",

        "output":
            str(output_file)
    }


    # ========================================================
    # REPRISE : SI LE FICHIER EXISTE DEJA
    # ========================================================

    if output_file.exists():

        try:

            existing = np.load(
                output_file
            )


            if (
                existing.ndim == 2
                and existing.shape[0] > 0
                and existing.shape[1] == 225
            ):

                result_row[
                    "frames"
                ] = existing.shape[0]

                result_row[
                    "success"
                ] = True

                result_row[
                    "status"
                ] = "ALREADY_VALID"

                print(
                    "  -> déjà extrait :",
                    existing.shape
                )

                rows.append(
                    result_row
                )

                continue

        except Exception:

            pass


    # ========================================================
    # VIDEO EXISTE ?
    # ========================================================

    if not video_path.exists():

        result_row[
            "status"
        ] = "VIDEO_MISSING"

        print(
            "  -> vidéo absente"
        )

        rows.append(
            result_row
        )

        continue


    # ========================================================
    # OUVERTURE
    # ========================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        result_row[
            "status"
        ] = "OPEN_FAILED"

        print(
            "  -> OpenCV impossible"
        )

        rows.append(
            result_row
        )

        continue


    # Reset du tracking entre les vidéos
    try:

        holistic.reset()

    except Exception:

        pass


    sequence = []

    pose_frames = 0
    left_frames = 0
    right_frames = 0
    any_hand_frames = 0


    # ========================================================
    # FRAME PAR FRAME
    # ========================================================

    while True:

        ok, frame = cap.read()

        if not ok:
            break


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # optimisation légère
        rgb.flags.writeable = False

        results = holistic.process(
            rgb
        )

        rgb.flags.writeable = True


        # ----------------------------------------------------
        # Détections
        # ----------------------------------------------------

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


        if pose_present:
            pose_frames += 1

        if left_present:
            left_frames += 1

        if right_present:
            right_frames += 1

        if (
            left_present
            or right_present
        ):

            any_hand_frames += 1


        # ----------------------------------------------------
        # 225 FEATURES
        # ----------------------------------------------------

        keypoints = extract_keypoints(
            results
        )


        sequence.append(
            keypoints
        )


    cap.release()


    # ========================================================
    # VALIDATION
    # ========================================================

    if len(sequence) == 0:

        result_row[
            "status"
        ] = "NO_FRAMES"

        print(
            "  -> aucune frame"
        )

        rows.append(
            result_row
        )

        continue


    sequence = np.stack(
        sequence
    ).astype(
        np.float32
    )


    if (
        sequence.ndim != 2
        or sequence.shape[1] != 225
    ):

        result_row[
            "status"
        ] = "INVALID_SHAPE"

        print(
            "  -> shape invalide :",
            sequence.shape
        )

        rows.append(
            result_row
        )

        continue


    # ========================================================
    # SAUVEGARDE
    # ========================================================

    np.save(
        output_file,
        sequence
    )


    result_row[
        "frames"
    ] = len(sequence)

    result_row[
        "pose_detected_frames"
    ] = pose_frames

    result_row[
        "left_hand_detected_frames"
    ] = left_frames

    result_row[
        "right_hand_detected_frames"
    ] = right_frames

    result_row[
        "any_hand_detected_frames"
    ] = any_hand_frames

    result_row[
        "success"
    ] = True

    result_row[
        "status"
    ] = "OK"


    print(
        "  ->",
        sequence.shape
    )


    rows.append(
        result_row
    )


# ============================================================
# FERMER MEDIAPIPE
# ============================================================

holistic.close()


# ============================================================
# RAPPORT
# ============================================================

report = pd.DataFrame(
    rows
)


report.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# RESULTATS
# ============================================================

print("\n")
print("=" * 85)
print("RÉSULTAT EXTRACTION KEYPOINTS")
print("=" * 85)


print(
    "Vidéos :",
    len(report)
)


print(
    "Succès :",
    int(
        report[
            "success"
        ].sum()
    )
)


print(
    "Échecs :",
    int(
        (
            ~report[
                "success"
            ]
        ).sum()
    )
)


print(
    "\nStatuts :"
)

print(
    report[
        "status"
    ]
    .value_counts()
)


success = report[
    report[
        "success"
    ]
].copy()


print(
    "\nNombre de frames :"
)

print(
    success[
        "frames"
    ]
    .describe()
)


print(
    "\nShape attendue pour chaque vidéo :"
)

print(
    "(nombre_de_frames, 225)"
)


print(
    "\nPar split :"
)

print(
    success[
        "split"
    ]
    .value_counts()
)


print(
    "\nDossier :"
)

print(
    OUTPUT_DIR
)


print(
    "\nRapport :"
)

print(
    REPORT_FILE
)
