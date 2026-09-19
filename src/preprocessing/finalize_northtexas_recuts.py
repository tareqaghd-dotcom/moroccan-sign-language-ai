from pathlib import Path
import subprocess

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

REPORT_CSV = (
    ROOT
    / "data"
    / "asl"
    / "northtexas_recut_report.csv"
)

INPUT_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "videos_northtexas_recut"
)

OUTPUT_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "videos_northtexas_final"
)

OUTPUT_REPORT = (
    ROOT
    / "data"
    / "asl"
    / "northtexas_final_report.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(REPORT_CSV)

rows = []


for _, row in df.iterrows():

    video = row["video"]

    src = INPUT_DIR / video
    dst = OUTPUT_DIR / video

    decoded_before = int(
        row["decoded_frames"]
    )

    trailing = int(
        row["trailing_dark_frames"]
    )

    # Nombre de frames que l'on souhaite conserver
    keep_frames = (
        decoded_before
        - trailing
    )


    result_row = {
        "video": video,
        "gloss": row["gloss"],
        "frames_before": decoded_before,
        "trailing_dark_removed": trailing,
        "expected_final_frames": keep_frames,
        "final_frames": 0,
        "final_fps": None,
        "last_frame_brightness": None,
        "success": False,
        "error": ""
    }


    if keep_frames <= 0:

        result_row["error"] = (
            "no_frames_after_trim"
        )

        rows.append(result_row)
        continue


    # ========================================================
    # SUPPRIMER UNIQUEMENT LES FRAMES SOMBRES FINALES
    # ========================================================

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",

        "-i",
        str(src),

        "-frames:v",
        str(keep_frames),

        "-c:v",
        "libx264",

        "-pix_fmt",
        "yuv420p",

        "-an",

        str(dst)
    ]


    process = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )


    if process.returncode != 0:

        result_row["error"] = (
            process.stderr[-500:]
        )

        rows.append(result_row)
        continue


    # ========================================================
    # VALIDATION OPENCV
    # ========================================================

    cap = cv2.VideoCapture(
        str(dst)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frames = 0
    last_brightness = None


    while True:

        ok, frame = cap.read()

        if not ok or frame is None:
            break

        frames += 1

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        last_brightness = float(
            gray.mean()
        )


    cap.release()


    result_row["final_frames"] = frames
    result_row["final_fps"] = fps
    result_row["last_frame_brightness"] = (
        last_brightness
    )


    result_row["success"] = (
        frames == keep_frames
        and frames > 0
    )


    if not result_row["success"]:

        result_row["error"] = (
            "frame_count_mismatch"
        )


    rows.append(result_row)


# ============================================================
# RAPPORT
# ============================================================

report = pd.DataFrame(rows)

report.to_csv(
    OUTPUT_REPORT,
    index=False
)


print("=" * 70)
print("FINALISATION NORTHTEXAS")
print("=" * 70)

print(
    "Vidéos :",
    len(report)
)

print(
    "Réussies :",
    int(report["success"].sum())
)

print(
    "Échecs :",
    int((~report["success"]).sum())
)

print(
    "\nFrames sombres supprimées au total :",
    int(
        report[
            "trailing_dark_removed"
        ].sum()
    )
)

print(
    "\nDernière frame encore sombre (< 35) :",
    int(
        (
            report[
                "last_frame_brightness"
            ] < 35
        ).sum()
    )
)

print(
    "\nFPS finaux :"
)

print(
    report[
        "final_fps"
    ]
    .round(3)
    .value_counts()
    .sort_index()
)

print(
    "\nDifférence entre attendu et obtenu :"
)

print(
    (
        report["final_frames"]
        - report["expected_final_frames"]
    )
    .value_counts()
    .sort_index()
)

print("\nRapport :")
print(OUTPUT_REPORT)

print("\nVidéos finales temporaires :")
print(OUTPUT_DIR)