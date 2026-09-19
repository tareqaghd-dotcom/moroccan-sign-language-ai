from pathlib import Path
from urllib.parse import urlparse, parse_qs
import json
import subprocess
import cv2


ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = ROOT / "data" / "asl" / "WLASL100.json"

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
    / "videos_recut_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TARGET_IDS = {
    "70212",
    "70323",
    "70016",
    "70245",
    "70356",
    "70378"
}


with open(
    JSON_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


for item in data:

    gloss = item["gloss"]

    for inst in item["instances"]:

        video_id = str(
            inst["video_id"]
        ).zfill(5)

        if video_id not in TARGET_IDS:
            continue


        # ====================================================
        # IDENTIFIANT YOUTUBE
        # ====================================================

        url = inst["url"]

        parsed = urlparse(url)

        youtube_id = parse_qs(
            parsed.query
        ).get("v", [None])[0]


        if youtube_id is None:

            print(
                video_id,
                "YouTube ID introuvable"
            )

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

            print(
                video_id,
                "source introuvable :",
                src
            )

            continue


        # ====================================================
        # TEMPS WLASL
        #
        # WLASL indexe à partir de 1.
        # Les frame_start/frame_end correspondent au FPS
        # indiqué dans les métadonnées.
        # ====================================================

        metadata_fps = float(
            inst.get("fps", 25)
        )

        frame_start = int(
            inst["frame_start"]
        )

        frame_end = int(
            inst["frame_end"]
        )


        start_seconds = (
            (frame_start - 1)
            / metadata_fps
        )


        duration_seconds = (
            (frame_end - frame_start + 1)
            / metadata_fps
        )


        # ====================================================
        # INFORMATIONS
        # ====================================================

        cap = cv2.VideoCapture(
            str(src)
        )

        actual_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        cap.release()


        print("\n======================================")
        print("video_id :", video_id)
        print("gloss :", gloss)
        print("FPS metadata :", metadata_fps)
        print("FPS source :", actual_fps)
        print(
            "Début corrigé :",
            round(start_seconds, 3),
            "s"
        )
        print(
            "Durée :",
            round(duration_seconds, 3),
            "s"
        )


        # ====================================================
        # FFMPEG
        #
        # Découpage temporel + normalisation à 25 FPS.
        #
        # On ne touche PAS aux vidéos originales.
        # ====================================================

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
            "fps=25",

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

            print("ERREUR FFmpeg")
            print(
                result.stderr[-1000:]
            )

            continue


        # ====================================================
        # CONTRÔLE DU FICHIER CRÉÉ
        # ====================================================

        cap = cv2.VideoCapture(
            str(dst)
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
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

        cap.release()


        print("Nouveau FPS :", fps)
        print("Frames :", frames)
        print(
            "Résolution :",
            width,
            "x",
            height
        )

        print(
            "Créé :",
            dst
        )