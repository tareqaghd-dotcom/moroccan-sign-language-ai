from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PREPROCESS_REPORT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "preprocess_report.csv"
)

VIDEOS_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "candidate_batch1_processed"
)

OUTPUT_REPORT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "video_quality_report.csv"
)

OUTPUT_CLEAN = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "technical_clean.csv"
)

OUTPUT_REVIEW = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "visual_review.csv"
)


# ============================================================
# BOOL
# ============================================================

def as_bool(series):

    return (
        series.astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )


# ============================================================
# CHARGEMENT
# ============================================================

df = pd.read_csv(
    PREPROCESS_REPORT,
    dtype={"video": str}
)

df["success_bool"] = as_bool(
    df["success"]
)

df = df[
    df["success_bool"]
].copy()


print("=" * 80)
print("CONTRÔLE QUALITÉ — CANDIDATE BATCH 1")
print("=" * 80)

print(
    "Vidéos à analyser :",
    len(df)
)


rows = []


# ============================================================
# BOUCLE
# ============================================================

for i, row in df.iterrows():

    video = row["video"]
    gloss = row["gloss"]

    path = (
        VIDEOS_DIR
        / video
    )


    print(
        f"[{len(rows)+1}/{len(df)}]",
        video,
        gloss
    )


    result = {

        "video":
            video,

        "gloss":
            gloss,

        "access_type":
            row.get("access_type"),

        "crop_mode":
            row.get("crop_mode"),

        "metadata_fps":
            row.get("metadata_fps"),

        "frame_start":
            row.get("frame_start"),

        "frame_end":
            row.get("frame_end"),

        "duration_requested":
            row.get(
                "duration_seconds_requested"
            ),

        "exists":
            False,

        "file_size_mb":
            0,

        "opened":
            False,

        "declared_frames":
            0,

        "decoded_frames":
            0,

        "fps":
            0,

        "width":
            0,

        "height":
            0,

        "duration":
            0,

        "brightness_mean":
            np.nan,

        "contrast_mean":
            np.nan,

        "blur_median":
            np.nan,

        "black_frame_ratio":
            np.nan,

        "freeze_ratio":
            np.nan,

        "motion_mean":
            np.nan,

        "expected_frames":
            np.nan,

        "frame_difference":
            np.nan,

        "technical_ok":
            False,

        "warning_black":
            False,

        "warning_blur":
            False,

        "warning_low_contrast":
            False,

        "warning_frozen":
            False,

        "warning_low_motion":
            False,

        "warning_short":
            False,

        "warning_long":
            False,

        "warning_crop_duration":
            False,

        "visual_warning_count":
            0,

        "needs_review":
            False,

        "error":
            ""
    }


    # ========================================================
    # EXISTENCE
    # ========================================================

    if not path.exists():

        result["error"] = "file_missing"

        rows.append(
            result
        )

        continue


    result["exists"] = True

    result["file_size_mb"] = (
        path.stat().st_size
        / 1024
        / 1024
    )


    # ========================================================
    # OUVERTURE
    # ========================================================

    cap = cv2.VideoCapture(
        str(path)
    )


    if not cap.isOpened():

        result["error"] = (
            "opencv_cannot_open"
        )

        rows.append(
            result
        )

        continue


    result["opened"] = True


    declared_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    result[
        "declared_frames"
    ] = declared_frames

    result[
        "fps"
    ] = fps

    result[
        "width"
    ] = width

    result[
        "height"
    ] = height


    # ========================================================
    # DECODE COMPLET
    # ========================================================

    brightness_values = []
    contrast_values = []
    blur_values = []

    black_frames = 0

    motion_values = []
    freeze_frames = 0

    previous_small_gray = None

    decoded = 0


    while True:

        ok, frame = cap.read()

        if not ok:
            break


        decoded += 1


        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # --------------------------------------------
        # Luminosité
        # --------------------------------------------

        brightness = float(
            gray.mean()
        )

        brightness_values.append(
            brightness
        )


        # --------------------------------------------
        # Contraste
        # --------------------------------------------

        contrast = float(
            gray.std()
        )

        contrast_values.append(
            contrast
        )


        # --------------------------------------------
        # Flou
        # --------------------------------------------

        blur = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        blur_values.append(
            blur
        )


        # --------------------------------------------
        # Frame noire
        # --------------------------------------------

        if brightness < 20:

            black_frames += 1


        # --------------------------------------------
        # Mouvement / freeze
        #
        # On réduit l'image pour rendre la mesure
        # comparable entre différentes résolutions.
        # --------------------------------------------

        small_gray = cv2.resize(
            gray,
            (160, 120)
        )


        if previous_small_gray is not None:

            difference = cv2.absdiff(
                small_gray,
                previous_small_gray
            )

            motion = float(
                difference.mean()
            )

            motion_values.append(
                motion
            )


            if motion < 1.0:

                freeze_frames += 1


        previous_small_gray = (
            small_gray
        )


    cap.release()


    result[
        "decoded_frames"
    ] = decoded


    # ========================================================
    # DUREE
    # ========================================================

    if fps > 0:

        duration = (
            decoded
            / fps
        )

    else:

        duration = 0


    result[
        "duration"
    ] = duration


    # ========================================================
    # METRIQUES
    # ========================================================

    if decoded > 0:

        result[
            "brightness_mean"
        ] = float(
            np.mean(
                brightness_values
            )
        )

        result[
            "contrast_mean"
        ] = float(
            np.mean(
                contrast_values
            )
        )

        result[
            "blur_median"
        ] = float(
            np.median(
                blur_values
            )
        )

        result[
            "black_frame_ratio"
        ] = (
            black_frames
            / decoded
        )


    if len(
        motion_values
    ) > 0:

        result[
            "motion_mean"
        ] = float(
            np.mean(
                motion_values
            )
        )

        result[
            "freeze_ratio"
        ] = (
            freeze_frames
            / len(
                motion_values
            )
        )


    # ========================================================
    # VERIFICATION DES CROPS
    # ========================================================

    duration_requested = row.get(
        "duration_seconds_requested"
    )


    try:

        duration_requested = float(
            duration_requested
        )

    except Exception:

        duration_requested = None


    if (
        row.get("crop_mode")
        == "METADATA_TIME_CROP"
        and duration_requested is not None
        and not np.isnan(
            duration_requested
        )
    ):

        expected_frames = round(
            duration_requested
            * 25
        )

        result[
            "expected_frames"
        ] = expected_frames

        result[
            "frame_difference"
        ] = (
            decoded
            - expected_frames
        )


        # différence de plus de 2 frames :
        # seulement un avertissement
        if abs(
            decoded
            - expected_frames
        ) > 2:

            result[
                "warning_crop_duration"
            ] = True


    # ========================================================
    # VALIDITE TECHNIQUE
    # ========================================================

    technical_ok = (
        decoded > 0
        and fps > 0
        and width > 0
        and height > 0
        and result[
            "file_size_mb"
        ] > 0.001
    )


    result[
        "technical_ok"
    ] = technical_ok


    if not technical_ok:

        result[
            "error"
        ] = "invalid_video_properties"


    # ========================================================
    # AVERTISSEMENTS VISUELS
    #
    # IMPORTANT :
    # ce ne sont PAS des critères de rejet automatique.
    # ========================================================

    if technical_ok:

        if (
            result[
                "black_frame_ratio"
            ] > 0.10
        ):

            result[
                "warning_black"
            ] = True


        if (
            not np.isnan(
                result[
                    "blur_median"
                ]
            )
            and result[
                "blur_median"
            ] < 20
        ):

            result[
                "warning_blur"
            ] = True


        if (
            not np.isnan(
                result[
                    "contrast_mean"
                ]
            )
            and result[
                "contrast_mean"
            ] < 20
        ):

            result[
                "warning_low_contrast"
            ] = True


        if (
            not np.isnan(
                result[
                    "freeze_ratio"
                ]
            )
            and result[
                "freeze_ratio"
            ] > 0.85
        ):

            result[
                "warning_frozen"
            ] = True


        if (
            not np.isnan(
                result[
                    "motion_mean"
                ]
            )
            and result[
                "motion_mean"
            ] < 1.0
        ):

            result[
                "warning_low_motion"
            ] = True


        if duration < 0.4:

            result[
                "warning_short"
            ] = True


        if duration > 12:

            result[
                "warning_long"
            ] = True


    warning_columns = [

        "warning_black",

        "warning_blur",

        "warning_low_contrast",

        "warning_frozen",

        "warning_low_motion",

        "warning_short",

        "warning_long",

        "warning_crop_duration"
    ]


    warning_count = sum(
        int(
            bool(
                result[col]
            )
        )
        for col
        in warning_columns
    )


    result[
        "visual_warning_count"
    ] = warning_count


    result[
        "needs_review"
    ] = (
        technical_ok
        and warning_count > 0
    )


    rows.append(
        result
    )


# ============================================================
# RAPPORT
# ============================================================

report = pd.DataFrame(
    rows
)


report.to_csv(
    OUTPUT_REPORT,
    index=False
)


technical_clean = report[
    report[
        "technical_ok"
    ]
].copy()


technical_clean.to_csv(
    OUTPUT_CLEAN,
    index=False
)


visual_review = report[
    report[
        "needs_review"
    ]
].copy()


visual_review.to_csv(
    OUTPUT_REVIEW,
    index=False
)


# ============================================================
# AFFICHAGE
# ============================================================

print("\n")
print("=" * 80)
print("RÉSULTAT CONTRÔLE TECHNIQUE")
print("=" * 80)


print(
    "Vidéos analysées :",
    len(report)
)


print(
    "Techniquement valides :",
    int(
        report[
            "technical_ok"
        ].sum()
    )
)


print(
    "Techniquement rejetées :",
    int(
        (
            ~report[
                "technical_ok"
            ]
        ).sum()
    )
)


print(
    "À revoir visuellement :",
    int(
        report[
            "needs_review"
        ].sum()
    )
)


print(
    "\n========== WARNINGS =========="
)


warning_columns = [

    "warning_black",

    "warning_blur",

    "warning_low_contrast",

    "warning_frozen",

    "warning_low_motion",

    "warning_short",

    "warning_long",

    "warning_crop_duration"
]


for col in warning_columns:

    print(
        f"{col}:",
        int(
            report[
                col
            ].sum()
        )
    )


print(
    "\nDurée des clips :"
)

print(
    report[
        "duration"
    ]
    .describe()
)


print(
    "\nFrames décodées :"
)

print(
    report[
        "decoded_frames"
    ]
    .describe()
)


print(
    "\nFPS :"
)

print(
    report[
        "fps"
    ]
    .round(3)
    .value_counts()
    .sort_index()
)


print(
    "\nRapport complet :"
)

print(
    OUTPUT_REPORT
)


print(
    "\nVidéos techniques propres :"
)

print(
    OUTPUT_CLEAN
)


print(
    "\nVidéos à examiner :"
)

print(
    OUTPUT_REVIEW
)
