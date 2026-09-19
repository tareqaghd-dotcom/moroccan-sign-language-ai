from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
import subprocess
import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = (
    ROOT
    / "data"
    / "asl"
    / "WLASL100.json"
)

RAW_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "raw_videos_mp4"
)

OUTPUT_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "videos_northtexas_recut"
)

REPORT_PATH = (
    ROOT
    / "data"
    / "asl"
    / "northtexas_recut_report.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHARGER WLASL
# ============================================================

with open(
    JSON_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


rows = []


# ============================================================
# TRAITER LES INSTANCES NORTHTEXAS
# ============================================================

for item in data:

    gloss = item["gloss"]

    for inst in item["instances"]:

        if inst.get("source") != "northtexas":
            continue


        video_id = str(
            inst["video_id"]
        ).zfill(5)

        url = inst["url"]


        # ----------------------------------------------------
        # YouTube ID
        # ----------------------------------------------------

        parsed = urlparse(url)

        youtube_id = parse_qs(
            parsed.query
        ).get("v", [None])[0]


        row = {
            "video": f"{video_id}.mp4",
            "gloss": gloss,
            "youtube_id": youtube_id,

            "frame_start": inst["frame_start"],
            "frame_end": inst["frame_end"],

            "metadata_fps": inst.get("fps", 25),

            "source_fps": None,

            "expected_frames": None,
            "decoded_frames": 0,

            "output_fps": None,

            "leading_dark_frames": None,
            "trailing_dark_frames": None,

            "success": False,
            "error": ""
        }


        if youtube_id is None:

            row["error"] = "youtube_id_not_found"
            rows.append(row)
            continue


        src = (
            RAW_DIR
            / f"{youtube_id}.mp4"
        )

        dst = (
            OUTPUT_DIR
            / f"{video_id}.mp4"
        )


        if not src.exists():

            row["error"] = "raw_source_missing"
            rows.append(row)
            continue


        # ----------------------------------------------------
        # FPS réel de la source
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            str(src)
        )

        source_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        cap.release()


        row["source_fps"] = source_fps


        # ----------------------------------------------------
        # CONVERSION FRAME WLASL -> TEMPS
        # ----------------------------------------------------

        metadata_fps = float(
            inst.get("fps", 25)
        )

        frame_start = int(
            inst["frame_start"]
        )

        frame_end = int(
            inst["frame_end"]
        )


        expected_frames = (
            frame_end
            - frame_start
            + 1
        )

        row["expected_frames"] = (
            expected_frames
        )


        start_seconds = (
            (frame_start - 1)
            / metadata_fps
        )

        duration_seconds = (
            expected_frames
            / metadata_fps
        )


        # ----------------------------------------------------
        # RECUT
        # ----------------------------------------------------

        command = [
            "ffmpeg",
            "-y",

            "-i",
            str(src),

            "-ss",
            str(start_seconds),

            "-t",
            str(duration_seconds),

            "-vf",
            f"fps={metadata_fps}",

            "-c:v",
            "libx264",

            "-pix_fmt",
            "yuv420p",

            "-an",

            str(dst)
        ]


        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )


        if result.returncode != 0:

            row["error"] = "ffmpeg_failed"

            rows.append(row)
            continue


        # ----------------------------------------------------
        # CONTRÔLE DU CLIP CRÉÉ
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            str(dst)
        )

        output_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        brightness = []


        while True:

            ok, frame = cap.read()

            if not ok or frame is None:
                break


            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            brightness.append(
                float(gray.mean())
            )


        cap.release()


        decoded_frames = len(
            brightness
        )


        row["decoded_frames"] = (
            decoded_frames
        )

        row["output_fps"] = (
            output_fps
        )


        # ----------------------------------------------------
        # FRAMES SOMBRES UNIQUEMENT AUX BORDURES
        #
        # On NE les supprime pas encore.
        # ----------------------------------------------------

        dark = [
            value < 35
            for value in brightness
        ]


        leading = 0

        for value in dark:

            if value:
                leading += 1

            else:
                break


        trailing = 0

        for value in reversed(dark):

            if value:
                trailing += 1

            else:
                break


        row["leading_dark_frames"] = (
            leading
        )

        row["trailing_dark_frames"] = (
            trailing
        )


        row["success"] = (
            decoded_frames > 0
        )


        rows.append(row)


        print(
            video_id,
            gloss,
            "| expected:",
            expected_frames,
            "| decoded:",
            decoded_frames,
            "| source FPS:",
            round(source_fps, 3),
            "| dark start/end:",
            leading,
            "/",
            trailing
        )


# ============================================================
# RAPPORT
# ============================================================

report = pd.DataFrame(rows)

report.to_csv(
    REPORT_PATH,
    index=False
)


print("\n")
print("=" * 65)
print("RÉSULTAT RECUT NORTHTEXAS")
print("=" * 65)

print(
    "Instances northtexas :",
    len(report)
)

print(
    "Recuts réussis :",
    int(report["success"].sum())
)

print(
    "Échecs :",
    int((~report["success"]).sum())
)


print(
    "\nNombre avec frames sombres au début :",
    int(
        (
            report["leading_dark_frames"]
            .fillna(0)
            > 0
        ).sum()
    )
)

print(
    "Nombre avec frames sombres à la fin :",
    int(
        (
            report["trailing_dark_frames"]
            .fillna(0)
            > 0
        ).sum()
    )
)


print(
    "\nDifférence expected/decoded :"
)

report["frame_difference"] = (
    report["decoded_frames"]
    - report["expected_frames"]
)

print(
    report["frame_difference"]
    .value_counts()
    .sort_index()
)


print("\nRapport :")
print(REPORT_PATH)

print("\nVidéos :")
print(OUTPUT_DIR)