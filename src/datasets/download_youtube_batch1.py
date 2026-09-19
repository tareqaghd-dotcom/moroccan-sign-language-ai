from pathlib import Path
import subprocess
import sys
import time
import json

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
    / "youtube_sources_unique.csv"
)

OUTPUT_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "candidate_batch1_raw"
    / "youtube"
)

GLOBAL_RAW_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "raw_videos_mp4"
)

REPORT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "youtube_download_report.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FFPROBE
# ============================================================

def probe_video(path):

    command = [
        "ffprobe",
        "-v",
        "error",

        "-select_streams",
        "v:0",

        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,duration",

        "-show_entries",
        "format=duration",

        "-of",
        "json",

        str(path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


    if result.returncode != 0:

        return {
            "valid": False,
            "codec": None,
            "width": None,
            "height": None,
            "fps": None,
            "duration": None,
            "error": result.stderr[-500:]
        }


    try:

        data = json.loads(
            result.stdout
        )

        streams = data.get(
            "streams",
            []
        )

        if not streams:

            return {
                "valid": False,
                "codec": None,
                "width": None,
                "height": None,
                "fps": None,
                "duration": None,
                "error": "no_video_stream"
            }


        stream = streams[0]


        # FPS sous forme 30000/1001 par exemple
        fps_string = stream.get(
            "avg_frame_rate",
            "0/1"
        )


        try:

            num, den = fps_string.split("/")

            num = float(num)
            den = float(den)

            fps = (
                num / den
                if den != 0
                else 0
            )

        except Exception:

            fps = None


        duration = stream.get(
            "duration"
        )


        if duration is None:

            duration = (
                data
                .get("format", {})
                .get("duration")
            )


        try:

            duration = float(
                duration
            )

        except Exception:

            duration = None


        return {

            "valid": True,

            "codec":
                stream.get(
                    "codec_name"
                ),

            "width":
                stream.get(
                    "width"
                ),

            "height":
                stream.get(
                    "height"
                ),

            "fps":
                fps,

            "duration":
                duration,

            "error":
                ""
        }


    except Exception as e:

        return {

            "valid": False,

            "codec":
                None,

            "width":
                None,

            "height":
                None,

            "fps":
                None,

            "duration":
                None,

            "error":
                str(e)
        }


# ============================================================
# CHARGER LES 211 SOURCES
# ============================================================

df = pd.read_csv(
    INPUT,
    dtype=str
)


print("=" * 80)
print("TÉLÉCHARGEMENT YOUTUBE — BATCH 1")
print("=" * 80)

print(
    "Sources YouTube uniques :",
    len(df)
)


rows = []


# ============================================================
# BOUCLE
# ============================================================

for index, row in df.iterrows():

    youtube_id = str(
        row["youtube_id"]
    ).strip()

    gloss = row.get(
        "gloss",
        ""
    )

    url = row.get(
        "url",
        ""
    )


    # URL canonique
    if (
        not isinstance(url, str)
        or not url.strip()
        or url == "nan"
    ):

        url = (
            "https://www.youtube.com/watch?v="
            + youtube_id
        )


    output = (
        OUTPUT_DIR
        / f"{youtube_id}.mp4"
    )

    global_output = (
        GLOBAL_RAW_DIR
        / f"{youtube_id}.mp4"
    )


    result_row = {

        "youtube_id":
            youtube_id,

        "gloss_example":
            gloss,

        "url":
            url,

        "status":
            "",

        "success":
            False,

        "path":
            "",

        "size_mb":
            None,

        "codec":
            None,

        "width":
            None,

        "height":
            None,

        "fps":
            None,

        "duration":
            None,

        "error":
            ""
    }


    print(
        f"\n[{index + 1}/{len(df)}]",
        youtube_id,
        "|",
        gloss
    )


    # ========================================================
    # 1. SOURCE GLOBALE DEJA DISPONIBLE
    # ========================================================

    if (
        global_output.exists()
        and global_output.stat().st_size > 1024
    ):

        info = probe_video(
            global_output
        )

        if info["valid"]:

            result_row[
                "status"
            ] = "GLOBAL_ALREADY_VALID"

            result_row[
                "success"
            ] = True

            result_row[
                "path"
            ] = str(
                global_output
            )

            result_row[
                "size_mb"
            ] = (
                global_output
                .stat()
                .st_size
                / 1024
                / 1024
            )

            for key in [
                "codec",
                "width",
                "height",
                "fps",
                "duration"
            ]:

                result_row[key] = (
                    info[key]
                )


            print(
                "  -> déjà disponible dans raw_videos_mp4"
            )

            rows.append(
                result_row
            )

            continue


    # ========================================================
    # 2. DEJA TELECHARGEE DANS LE BATCH
    # ========================================================

    if (
        output.exists()
        and output.stat().st_size > 1024
    ):

        info = probe_video(
            output
        )

        if info["valid"]:

            result_row[
                "status"
            ] = "ALREADY_VALID"

            result_row[
                "success"
            ] = True

            result_row[
                "path"
            ] = str(
                output
            )

            result_row[
                "size_mb"
            ] = (
                output
                .stat()
                .st_size
                / 1024
                / 1024
            )

            for key in [
                "codec",
                "width",
                "height",
                "fps",
                "duration"
            ]:

                result_row[key] = (
                    info[key]
                )


            print(
                "  -> déjà téléchargée et valide"
            )

            rows.append(
                result_row
            )

            continue


    # ========================================================
    # 3. TELECHARGEMENT YT-DLP
    # ========================================================

    output_template = str(
        OUTPUT_DIR
        / "%(id)s.%(ext)s"
    )


    command = [

        sys.executable,
        "-m",
        "yt_dlp",

        "--no-playlist",

        "--retries",
        "3",

        "--fragment-retries",
        "3",

        "--socket-timeout",
        "25",

        "--no-part",

        "--remux-video",
        "mp4",

        "-f",
        (
            "bestvideo[ext=mp4][height<=720]"
            "/best[ext=mp4][height<=720]"
            "/bestvideo[height<=720]"
            "/best[height<=720]"
            "/best"
        ),

        "-o",
        output_template,

        url
    ]


    try:

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180
        )


        # ----------------------------------------------------
        # yt-dlp a échoué
        # ----------------------------------------------------

        if process.returncode != 0:

            result_row[
                "status"
            ] = "YTDLP_FAILED"

            result_row[
                "error"
            ] = (
                process.stderr[-1000:]
            )

            print(
                "  -> yt-dlp échec"
            )

            rows.append(
                result_row
            )

            continue


        # ----------------------------------------------------
        # RETROUVER LE FICHIER PRODUIT
        # ----------------------------------------------------

        candidates = list(
            OUTPUT_DIR.glob(
                f"{youtube_id}.*"
            )
        )


        candidates = [
            p
            for p in candidates
            if p.is_file()
            and p.suffix not in [
                ".part",
                ".ytdl",
                ".json"
            ]
        ]


        if not candidates:

            result_row[
                "status"
            ] = "FILE_NOT_FOUND_AFTER_DOWNLOAD"

            result_row[
                "error"
            ] = (
                "yt-dlp terminé mais aucun fichier trouvé"
            )

            print(
                "  -> aucun fichier produit"
            )

            rows.append(
                result_row
            )

            continue


        # Prendre le plus gros fichier
        downloaded = max(
            candidates,
            key=lambda p:
                p.stat().st_size
        )


        # ----------------------------------------------------
        # SI PAS MP4 : CONVERSION CONTENEUR MP4
        # ----------------------------------------------------

        if downloaded.suffix.lower() != ".mp4":

            converted = (
                OUTPUT_DIR
                / f"{youtube_id}.mp4"
            )


            ffmpeg_command = [

                "ffmpeg",
                "-y",

                "-i",
                str(downloaded),

                "-map",
                "0:v:0",

                "-c:v",
                "copy",

                "-an",

                str(converted)
            ]


            ffmpeg_result = (
                subprocess.run(
                    ffmpeg_command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True
                )
            )


            if (
                ffmpeg_result.returncode
                == 0
                and converted.exists()
            ):

                downloaded.unlink(
                    missing_ok=True
                )

                downloaded = (
                    converted
                )


        # ----------------------------------------------------
        # VALIDATION FINALE
        # ----------------------------------------------------

        info = probe_video(
            downloaded
        )


        if not info["valid"]:

            result_row[
                "status"
            ] = "INVALID_VIDEO"

            result_row[
                "error"
            ] = info[
                "error"
            ]

            downloaded.unlink(
                missing_ok=True
            )

            print(
                "  -> fichier téléchargé invalide"
            )

            rows.append(
                result_row
            )

            continue


        # ----------------------------------------------------
        # SUCCES
        # ----------------------------------------------------

        result_row[
            "status"
        ] = "DOWNLOADED_VALID"

        result_row[
            "success"
        ] = True

        result_row[
            "path"
        ] = str(
            downloaded
        )

        result_row[
            "size_mb"
        ] = (
            downloaded
            .stat()
            .st_size
            / 1024
            / 1024
        )


        for key in [
            "codec",
            "width",
            "height",
            "fps",
            "duration"
        ]:

            result_row[key] = (
                info[key]
            )


        print(
            "  -> OK",
            round(
                result_row[
                    "size_mb"
                ],
                2
            ),
            "MB",
            "| FPS:",
            round(
                info["fps"],
                3
            )
            if info["fps"]
            else "?"
        )


    except subprocess.TimeoutExpired:

        result_row[
            "status"
        ] = "TIMEOUT"

        result_row[
            "error"
        ] = (
            "yt-dlp timeout > 180 s"
        )

        print(
            "  -> timeout"
        )


    except Exception as e:

        result_row[
            "status"
        ] = "EXCEPTION"

        result_row[
            "error"
        ] = str(e)[:1000]

        print(
            "  -> exception :",
            e
        )


    rows.append(
        result_row
    )


    # légère pause
    time.sleep(
        0.3
    )


# ============================================================
# RAPPORT
# ============================================================

new_report = pd.DataFrame(
    rows
)


if REPORT.exists():

    old = pd.read_csv(
        REPORT,
        dtype=str
    )

    old = old[
        ~old[
            "youtube_id"
        ].isin(
            new_report[
                "youtube_id"
            ]
        )
    ]

    final_report = pd.concat(
        [
            old,
            new_report
        ],
        ignore_index=True
    )

else:

    final_report = (
        new_report
    )


final_report.to_csv(
    REPORT,
    index=False
)


# ============================================================
# RESULTAT
# ============================================================

print("\n")
print("=" * 80)
print("RÉSULTAT YOUTUBE")
print("=" * 80)


print(
    "Sources traitées :",
    len(new_report)
)


print(
    "Succès :",
    int(
        new_report[
            "success"
        ].astype(bool)
        .sum()
    )
)


print(
    "Échecs :",
    int(
        (
            ~new_report[
                "success"
            ].astype(bool)
        ).sum()
    )
)


print(
    "\nStatuts :"
)

print(
    new_report[
        "status"
    ]
    .value_counts(
        dropna=False
    )
)


successful = new_report[
    new_report[
        "success"
    ].astype(bool)
].copy()


if len(successful) > 0:

    successful[
        "fps"
    ] = pd.to_numeric(
        successful[
            "fps"
        ],
        errors="coerce"
    )


    print(
        "\nFPS téléchargés :"
    )

    print(
        successful[
            "fps"
        ]
        .round(3)
        .value_counts()
        .sort_index()
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
    REPORT
)
