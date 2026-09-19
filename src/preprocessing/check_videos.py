from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "data" / "asl" / "asl_selected.csv"

VIDEOS_DIR = Path.home() / "WLASL" / "start_kit" / "videos"

REPORT_PATH = ROOT / "data" / "asl" / "video_quality_report.csv"

TECHNICAL_CLEAN_PATH = (
    ROOT / "data" / "asl" / "asl_clean_technical.csv"
)

REVIEW_PATH = (
    ROOT / "data" / "asl" / "videos_to_review.csv"
)


# ============================================================
# 2. VÉRIFICATION DES CHEMINS
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV introuvable : {CSV_PATH}"
    )

if not VIDEOS_DIR.exists():
    raise FileNotFoundError(
        f"Dossier vidéos introuvable : {VIDEOS_DIR}"
    )


# ============================================================
# 3. FFmpeg
# ============================================================

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

print("FFmpeg disponible :", FFMPEG_AVAILABLE)


# ============================================================
# 4. CHARGER LE DATASET
# ============================================================

df = pd.read_csv(
    CSV_PATH,
    dtype={"video": str}
)

print("Nombre de vidéos dans asl_selected.csv :", len(df))
print("Nombre de classes :", df["gloss"].nunique())


# ============================================================
# 5. TEST COMPLET FFmpeg
# ============================================================

def check_with_ffmpeg(path):

    if not FFMPEG_AVAILABLE:
        return None, ""

    command = [
        "ffmpeg",
        "-v", "error",
        "-xerror",
        "-i", str(path),
        "-f", "null",
        "-"
    ]

    try:

        process = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )

        error_message = process.stderr.strip()

        return process.returncode == 0, error_message[:500]

    except subprocess.TimeoutExpired:

        return False, "FFmpeg timeout"

    except Exception as error:

        return False, str(error)[:500]


# ============================================================
# 6. ANALYSER UNE VIDÉO
# ============================================================

def inspect_video(filename):

    path = VIDEOS_DIR / filename

    result = {

        "video": filename,

        # fichier
        "exists": False,
        "file_size_mb": 0.0,

        # ouverture
        "opened": False,

        # métadonnées
        "declared_frames": 0,
        "decoded_frames": 0,
        "decode_ratio": np.nan,

        "fps": 0.0,

        "width": 0,
        "height": 0,

        "duration": 0.0,

        # qualité visuelle
        "brightness_mean": np.nan,
        "contrast_mean": np.nan,
        "blur_median": np.nan,

        "black_frame_ratio": np.nan,
        "freeze_ratio": np.nan,
        "motion_mean": np.nan,

        # FFmpeg
        "ffmpeg_ok": None,
        "ffmpeg_error": "",

        # résultat technique
        "technical_ok": False,

        # raison éventuelle du rejet
        "reject_reason": ""
    }


    # ========================================================
    # A. Le fichier existe-t-il ?
    # ========================================================

    if not path.is_file():

        result["reject_reason"] = "missing_file"

        return result


    result["exists"] = True

    result["file_size_mb"] = (
        path.stat().st_size / (1024 * 1024)
    )


    # ========================================================
    # B. Test FFmpeg
    # ========================================================

    ffmpeg_ok, ffmpeg_error = check_with_ffmpeg(path)

    result["ffmpeg_ok"] = ffmpeg_ok
    result["ffmpeg_error"] = ffmpeg_error


    # ========================================================
    # C. Ouvrir avec OpenCV
    # ========================================================

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():

        result["reject_reason"] = "opencv_cannot_open"

        cap.release()

        return result


    result["opened"] = True


    # ========================================================
    # D. Lire les métadonnées
    # ========================================================

    declared_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )


    result["declared_frames"] = declared_frames
    result["fps"] = fps
    result["width"] = width
    result["height"] = height


    # ========================================================
    # E. LIRE TOUTES LES FRAMES
    # ========================================================

    decoded_frames = 0

    brightness_values = []
    contrast_values = []
    blur_values = []
    motion_values = []

    black_frames = 0
    frozen_frames = 0

    previous_small_gray = None


    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame is None:
            break

        if frame.size == 0:
            break


        decoded_frames += 1


        # ----------------------------------------------------
        # Passage en niveaux de gris
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )


        # ----------------------------------------------------
        # Luminosité
        # ----------------------------------------------------

        brightness = float(
            np.mean(gray)
        )

        brightness_values.append(brightness)


        # ----------------------------------------------------
        # Contraste
        # ----------------------------------------------------

        contrast = float(
            np.std(gray)
        )

        contrast_values.append(contrast)


        # ----------------------------------------------------
        # Flou
        #
        # variance du Laplacien :
        # valeur faible = image potentiellement floue
        # ----------------------------------------------------

        blur_score = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F
            ).var()
        )

        blur_values.append(blur_score)


        # ----------------------------------------------------
        # Frame presque noire
        #
        # Pour l'instant on MESURE seulement.
        # On ne supprime pas automatiquement.
        # ----------------------------------------------------

        if brightness < 8:
            black_frames += 1


        # ----------------------------------------------------
        # Mouvement / frames figées
        #
        # On réduit l'image pour accélérer la comparaison.
        # ----------------------------------------------------

        small_gray = cv2.resize(
            gray,
            (64, 64)
        )


        if previous_small_gray is not None:

            difference = cv2.absdiff(
                small_gray,
                previous_small_gray
            )

            motion_score = float(
                np.mean(difference)
            )

            motion_values.append(motion_score)


            # Très peu de différence entre deux frames
            if motion_score < 1.0:
                frozen_frames += 1


        previous_small_gray = small_gray


    cap.release()


    # ========================================================
    # F. STATISTIQUES APRÈS LECTURE COMPLÈTE
    # ========================================================

    result["decoded_frames"] = decoded_frames


    if declared_frames > 0:

        result["decode_ratio"] = (
            decoded_frames / declared_frames
        )


    if fps > 0:

        result["duration"] = (
            decoded_frames / fps
        )


    if decoded_frames > 0:

        result["brightness_mean"] = float(
            np.mean(brightness_values)
        )

        result["contrast_mean"] = float(
            np.mean(contrast_values)
        )

        result["blur_median"] = float(
            np.median(blur_values)
        )

        result["black_frame_ratio"] = (
            black_frames / decoded_frames
        )


    if decoded_frames > 1:

        result["freeze_ratio"] = (
            frozen_frames /
            (decoded_frames - 1)
        )


    if motion_values:

        result["motion_mean"] = float(
            np.mean(motion_values)
        )


    # ========================================================
    # G. DÉCISION TECHNIQUE
    # ========================================================

    reasons = []


    if decoded_frames == 0:

        reasons.append("zero_decoded_frames")


    if width <= 0 or height <= 0:

        reasons.append("invalid_resolution")


    # FFmpeg False = erreur réelle trouvée
    #
    # None signifie simplement que FFmpeg n'était
    # pas installé : ce n'est pas une raison de rejet.
    if ffmpeg_ok is False:

        reasons.append("ffmpeg_decode_error")


    # Attention :
    # On ne rejette PAS encore automatiquement pour :
    #
    # - FPS faible
    # - vidéo courte
    # - flou
    # - sombre
    # - peu de mouvement
    # - résolution faible
    #
    # Ces critères doivent être analysés sur
    # l'ensemble du dataset avant de fixer des seuils.


    if len(reasons) == 0:

        result["technical_ok"] = True

    else:

        result["reject_reason"] = ";".join(reasons)


    return result


# ============================================================
# 7. ANALYSER LES 1137 VIDÉOS
# ============================================================

results = []


for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Analyse complète des vidéos"
):

    info = inspect_video(
        row["video"]
    )

    info["gloss"] = row["gloss"]
    info["split"] = row["split"]

    results.append(info)


# ============================================================
# 8. CRÉER LE RAPPORT COMPLET
# ============================================================

report = pd.DataFrame(results)


columns = [

    "video",
    "gloss",
    "split",

    "exists",
    "file_size_mb",

    "opened",

    "declared_frames",
    "decoded_frames",
    "decode_ratio",

    "fps",

    "width",
    "height",

    "duration",

    "brightness_mean",
    "contrast_mean",
    "blur_median",

    "black_frame_ratio",
    "freeze_ratio",
    "motion_mean",

    "ffmpeg_ok",
    "ffmpeg_error",

    "technical_ok",
    "reject_reason"
]


report = report[columns]


# Arrondir seulement pour rendre le CSV lisible

numeric_columns = [
    "file_size_mb",
    "decode_ratio",
    "fps",
    "duration",
    "brightness_mean",
    "contrast_mean",
    "blur_median",
    "black_frame_ratio",
    "freeze_ratio",
    "motion_mean"
]


report[numeric_columns] = (
    report[numeric_columns].round(4)
)


report.to_csv(
    REPORT_PATH,
    index=False
)


# ============================================================
# 9. DATASET TECHNIQUEMENT PROPRE
# ============================================================

technical_videos = set(

    report.loc[
        report["technical_ok"] == True,
        "video"
    ]

)


technical_clean = df[
    df["video"].isin(technical_videos)
].copy()


technical_clean.to_csv(
    TECHNICAL_CLEAN_PATH,
    index=False
)


# ============================================================
# 10. REPÉRER LES VIDÉOS À EXAMINER
# ============================================================

review = report[
    report["technical_ok"] == True
].copy()


# ------------------------------------------------------------
# Decode ratio faible
# ------------------------------------------------------------

review["warning_decode"] = (

    review["decode_ratio"].notna()

    &

    (review["decode_ratio"] < 0.95)

)


# ------------------------------------------------------------
# Beaucoup de frames noires
# ------------------------------------------------------------

review["warning_black"] = (

    review["black_frame_ratio"].notna()

    &

    (review["black_frame_ratio"] > 0.20)

)


# ------------------------------------------------------------
# Beaucoup de frames quasiment identiques
# ------------------------------------------------------------

review["warning_frozen"] = (

    review["freeze_ratio"].notna()

    &

    (review["freeze_ratio"] > 0.80)

)


# ============================================================
# 11. DÉTECTION D'OUTLIERS BASÉE SUR NOTRE DATASET
# ============================================================

# On utilise les distributions de NOS vidéos.
#
# Ces seuils servent uniquement à envoyer des vidéos
# en revue.
#
# Ils ne servent PAS encore à les supprimer.


if len(review) > 0:

    blur_limit = review["blur_median"].quantile(0.05)

    contrast_limit = review[
        "contrast_mean"
    ].quantile(0.05)

    motion_limit = review[
        "motion_mean"
    ].quantile(0.05)


    review["warning_blur"] = (
        review["blur_median"] < blur_limit
    )

    review["warning_low_contrast"] = (
        review["contrast_mean"] < contrast_limit
    )

    review["warning_low_motion"] = (
        review["motion_mean"] < motion_limit
    )


else:

    review["warning_blur"] = False
    review["warning_low_contrast"] = False
    review["warning_low_motion"] = False


warning_columns = [

    "warning_decode",
    "warning_black",
    "warning_frozen",
    "warning_blur",
    "warning_low_contrast",
    "warning_low_motion"

]


review["review_needed"] = (
    review[warning_columns].any(axis=1)
)


videos_to_review = review[
    review["review_needed"] == True
].copy()


videos_to_review.to_csv(
    REVIEW_PATH,
    index=False
)


# ============================================================
# 12. AFFICHER LE RÉSULTAT
# ============================================================

print("\n")
print("=" * 60)
print("RÉSULTAT DU CLEANING TECHNIQUE")
print("=" * 60)


print(
    "Vidéos analysées :",
    len(report)
)


print(
    "Vidéos techniquement valides :",
    int(report["technical_ok"].sum())
)


print(
    "Vidéos techniquement rejetées :",
    int((~report["technical_ok"]).sum())
)


print(
    "Vidéos à examiner pour la qualité :",
    len(videos_to_review)
)


print(
    "Classes restantes après contrôle technique :",
    technical_clean["gloss"].nunique()
)


print("\nFichiers créés :")

print(REPORT_PATH)
print(TECHNICAL_CLEAN_PATH)
print(REVIEW_PATH)


print("\nRaisons des rejets techniques :")

invalid = report[
    report["technical_ok"] == False
]

if len(invalid) == 0:

    print("Aucun rejet technique.")

else:

    print(
        invalid[
            [
                "video",
                "gloss",
                "reject_reason"
            ]
        ].to_string(index=False)
    )


print("\nStatistiques générales :")

print(
    report[
        report["technical_ok"] == True
    ][
        [
            "decoded_frames",
            "fps",
            "width",
            "height",
            "duration",
            "brightness_mean",
            "contrast_mean",
            "blur_median",
            "black_frame_ratio",
            "freeze_ratio",
            "motion_mean"
        ]
    ].describe()
)