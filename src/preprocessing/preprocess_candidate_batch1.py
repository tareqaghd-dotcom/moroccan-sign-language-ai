from pathlib import Path
import subprocess
import math

import pandas as pd


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "availability_after_downloads.csv"
)

DIRECT_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "candidate_batch1_raw"
    / "direct_mp4"
)

YOUTUBE_NEW_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "candidate_batch1_raw"
    / "youtube"
)

YOUTUBE_OLD_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "raw_videos_mp4"
)

OUTPUT_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "candidate_batch1_processed"
)

REPORT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "preprocess_report.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTILS
# ============================================================

def safe_float(value):

    try:

        value = float(value)

        if math.isnan(value):
            return None

        return value

    except Exception:

        return None


def safe_int(value):

    try:

        value = float(value)

        if math.isnan(value):
            return None

        return int(value)

    except Exception:

        return None


def ffprobe_video(path):

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1",
        str(path)
    ]


    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


    if result.returncode != 0:

        return False, {}


    values = {}


    for line in result.stdout.splitlines():

        if "=" not in line:
            continue

        key, value = line.split(
            "=",
            1
        )

        values[key] = value


    return True, values


def full_decode_ok(path):

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-"
    ]


    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )


    return (
        result.returncode == 0,
        result.stderr[-500:]
    )


# ============================================================
# CHARGER LES INSTANCES ACCESSIBLES
# ============================================================

df = pd.read_csv(
    INPUT,
    dtype={
        "video": str,
        "youtube_id": str
    }
)


df = df[
    df["source_available_now"]
    .astype(str)
    .str.lower()
    .isin(["true", "1"])
].copy()


print("=" * 80)
print("PREPROCESSING CANDIDATE BATCH 1")
print("=" * 80)

print(
    "Instances accessibles :",
    len(df)
)


print(
    "\nTypes d'accès :"
)

print(
    df["access_type"]
    .value_counts()
)


rows = []


# ============================================================
# BOUCLE
# ============================================================

for position, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    video = str(
        row["video"]
    )

    gloss = row["gloss"]

    access_type = row[
        "access_type"
    ]

    youtube_id = row.get(
        "youtube_id"
    )


    if (
        pd.isna(youtube_id)
        or str(youtube_id) == "nan"
    ):

        youtube_id = None

    else:

        youtube_id = str(
            youtube_id
        )


    frame_start = safe_int(
        row.get(
            "frame_start"
        )
    )

    frame_end = safe_int(
        row.get(
            "frame_end"
        )
    )

    metadata_fps = safe_float(
        row.get(
            "metadata_fps"
        )
    )


    output = (
        OUTPUT_DIR
        / video
    )


    result_row = {

        "video":
            video,

        "gloss":
            gloss,

        "access_type":
            access_type,

        "youtube_id":
            youtube_id,

        "frame_start":
            frame_start,

        "frame_end":
            frame_end,

        "metadata_fps":
            metadata_fps,

        "source_path":
            "",

        "crop_mode":
            "",

        "start_seconds":
            None,

        "duration_seconds_requested":
            None,

        "success":
            False,

        "status":
            "",

        "decode_ok":
            False,

        "error":
            ""
    }


    print(
        f"\n[{position}/{len(df)}]",
        video,
        gloss,
        "|",
        access_type
    )


    # ========================================================
    # TROUVER LA SOURCE
    # ========================================================

    if access_type == "DIRECT_CLIP":

        source_path = (
            DIRECT_DIR
            / video
        )


    elif access_type == "YOUTUBE_NEW":

        if youtube_id is None:

            result_row[
                "status"
            ] = "NO_YOUTUBE_ID"

            rows.append(
                result_row
            )

            print(
                "  -> youtube_id manquant"
            )

            continue


        source_path = (
            YOUTUBE_NEW_DIR
            / f"{youtube_id}.mp4"
        )


    elif access_type == "RAW_PREEXISTING":

        if youtube_id is None:

            result_row[
                "status"
            ] = "NO_YOUTUBE_ID"

            rows.append(
                result_row
            )

            print(
                "  -> youtube_id manquant"
            )

            continue


        source_path = (
            YOUTUBE_OLD_DIR
            / f"{youtube_id}.mp4"
        )


    else:

        result_row[
            "status"
        ] = "UNKNOWN_ACCESS_TYPE"

        rows.append(
            result_row
        )

        print(
            "  -> type inconnu"
        )

        continue


    result_row[
        "source_path"
    ] = str(
        source_path
    )


    # ========================================================
    # SOURCE ABSENTE
    # ========================================================

    if not source_path.exists():

        result_row[
            "status"
        ] = "SOURCE_MISSING"

        rows.append(
            result_row
        )

        print(
            "  -> fichier source absent"
        )

        continue


    # ========================================================
    # SI LA SORTIE EXISTE DEJA
    # ========================================================

    if (
        output.exists()
        and output.stat().st_size > 1024
    ):

        probe_ok, _ = ffprobe_video(
            output
        )

        decode_ok, decode_error = (
            full_decode_ok(
                output
            )
        )


        if (
            probe_ok
            and decode_ok
        ):

            result_row[
                "success"
            ] = True

            result_row[
                "decode_ok"
            ] = True

            result_row[
                "status"
            ] = "ALREADY_VALID"

            print(
                "  -> déjà traité et valide"
            )

            rows.append(
                result_row
            )

            continue


    # ========================================================
    # DECISION DE DECOUPAGE
    # ========================================================

    #
    # IMPORTANT :
    #
    # DIRECT_MP4 :
    # le lien représente déjà le clip de l'instance.
    #
    # YOUTUBE :
    # le fichier est une source longue.
    # On convertit les indices des métadonnées en TEMPS.
    #
    # start_seconds =
    # (frame_start - 1) / metadata_fps
    #
    # duration =
    # nombre_de_frames / metadata_fps
    #


    if access_type == "DIRECT_CLIP":

        crop_mode = (
            "FULL_DIRECT_CLIP"
        )

        start_seconds = None
        duration_seconds = None


    else:

        if (
            frame_start is not None
            and frame_end is not None
            and metadata_fps is not None
            and metadata_fps > 0
            and frame_end > 0
            and frame_start > 0
            and frame_end >= frame_start
        ):

            crop_mode = (
                "METADATA_TIME_CROP"
            )


            start_seconds = (
                (frame_start - 1)
                / metadata_fps
            )


            frame_count = (
                frame_end
                - frame_start
                + 1
            )


            duration_seconds = (
                frame_count
                / metadata_fps
            )


        else:

            crop_mode = (
                "FULL_SOURCE"
            )

            start_seconds = None
            duration_seconds = None


    result_row[
        "crop_mode"
    ] = crop_mode

    result_row[
        "start_seconds"
    ] = start_seconds

    result_row[
        "duration_seconds_requested"
    ] = duration_seconds


    # ========================================================
    # FFMPEG
    # ========================================================

    temp = output.with_suffix(
        ".tmp.mp4"
    )


    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(source_path)
    ]


    # Recherche précise APRES -i
    if start_seconds is not None:

        command += [
            "-ss",
            f"{start_seconds:.6f}"
        ]


    if duration_seconds is not None:

        command += [
            "-t",
            f"{duration_seconds:.6f}"
        ]


    command += [

        "-map",
        "0:v:0",

        "-an",

        "-vf",
        "fps=25",

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "18",

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        str(temp)
    ]


    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )


    # ========================================================
    # FFMPEG ECHEC
    # ========================================================

    if (
        result.returncode != 0
        or not temp.exists()
    ):

        temp.unlink(
            missing_ok=True
        )

        result_row[
            "status"
        ] = "FFMPEG_FAILED"

        result_row[
            "error"
        ] = result.stderr[-500:]

        print(
            "  -> FFmpeg échec"
        )

        rows.append(
            result_row
        )

        continue


    # ========================================================
    # TAILLE
    # ========================================================

    if temp.stat().st_size < 1024:

        temp.unlink(
            missing_ok=True
        )

        result_row[
            "status"
        ] = "OUTPUT_TOO_SMALL"

        print(
            "  -> sortie trop petite"
        )

        rows.append(
            result_row
        )

        continue


    # ========================================================
    # FFPROBE
    # ========================================================

    probe_ok, probe_info = (
        ffprobe_video(
            temp
        )
    )


    if not probe_ok:

        temp.unlink(
            missing_ok=True
        )

        result_row[
            "status"
        ] = "FFPROBE_FAILED"

        print(
            "  -> ffprobe échec"
        )

        rows.append(
            result_row
        )

        continue


    # ========================================================
    # DECODE COMPLET
    # ========================================================

    decode_ok, decode_error = (
        full_decode_ok(
            temp
        )
    )


    result_row[
        "decode_ok"
    ] = decode_ok


    if not decode_ok:

        temp.unlink(
            missing_ok=True
        )

        result_row[
            "status"
        ] = "DECODE_FAILED"

        result_row[
            "error"
        ] = decode_error

        print(
            "  -> erreur de décodage"
        )

        rows.append(
            result_row
        )

        continue


    # ========================================================
    # SUCCES
    # ========================================================

    temp.replace(
        output
    )


    result_row[
        "success"
    ] = True

    result_row[
        "status"
    ] = "PROCESSED_VALID"


    print(
        "  -> OK",
        crop_mode
    )


    rows.append(
        result_row
    )


# ============================================================
# RAPPORT
# ============================================================

report = pd.DataFrame(
    rows
)


report.to_csv(
    REPORT,
    index=False
)


# ============================================================
# RÉSULTAT
# ============================================================

print("\n")
print("=" * 80)
print("RÉSULTAT PREPROCESSING")
print("=" * 80)


print(
    "Instances traitées :",
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


print(
    "\nModes de découpage :"
)

print(
    report[
        "crop_mode"
    ]
    .value_counts()
)


print(
    "\nPar type d'accès :"
)

print(
    report
    .groupby(
        "access_type"
    )
    .agg(
        total=(
            "video",
            "count"
        ),

        success=(
            "success",
            "sum"
        )
    )
    .to_string()
)


print(
    "\nDossier clips :"
)

print(
    OUTPUT_DIR
)


print(
    "\nRapport :"
)

print(
    REPORT
)
