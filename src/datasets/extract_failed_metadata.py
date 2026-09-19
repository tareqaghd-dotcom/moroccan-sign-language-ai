from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

WLASL_JSON = ROOT / "data" / "asl" / "WLASL100.json"
REPORT_CSV = ROOT / "data" / "asl" / "video_quality_report.csv"

OUTPUT_CSV = ROOT / "data" / "asl" / "failed_videos_metadata.csv"


# ============================================================
# 1. Charger le rapport de cleaning
# ============================================================

report = pd.read_csv(
    REPORT_CSV,
    dtype={"video": str}
)


# Garder uniquement les vidéos rejetées
failed = report[
    report["technical_ok"] == False
].copy()


print("Nombre de vidéos rejetées :", len(failed))


# ============================================================
# 2. Transformer 07099.mp4 -> 07099
# ============================================================

failed["video_id"] = (
    failed["video"]
    .str.replace(".mp4", "", regex=False)
)


failed_ids = set(failed["video_id"])


# ============================================================
# 3. Charger WLASL100.json
# ============================================================

with open(
    WLASL_JSON,
    "r",
    encoding="utf-8"
) as f:

    wlasl = json.load(f)


# ============================================================
# 4. Rechercher les métadonnées des vidéos rejetées
# ============================================================

rows = []


for class_data in wlasl:

    gloss = class_data["gloss"]

    for instance in class_data["instances"]:

        video_id = str(instance["video_id"])

        # préserver les zéros au début
        video_id = video_id.zfill(5)

        if video_id not in failed_ids:
            continue


        row = {
            "video_id": video_id,
            "video": f"{video_id}.mp4",
            "gloss": gloss,

            "url": instance.get("url"),
            "source": instance.get("source"),

            "signer_id": instance.get("signer_id"),

            "frame_start": instance.get("frame_start"),
            "frame_end": instance.get("frame_end"),

            "fps": instance.get("fps"),

            "split": instance.get("split"),

            "variation_id": instance.get("variation_id"),
            "instance_id": instance.get("instance_id")
        }

        rows.append(row)


# ============================================================
# 5. Créer le CSV
# ============================================================

metadata = pd.DataFrame(rows)


metadata.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# 6. Résumé
# ============================================================

print("\nMétadonnées retrouvées :", len(metadata))

print(
    "Vidéo IDs différentes :",
    metadata["video_id"].nunique()
)

print(
    "Classes concernées :",
    metadata["gloss"].nunique()
)


print("\nRépartition par split :")

print(
    metadata["split"].value_counts(
        dropna=False
    )
)


print("\nNombre de signants :")

print(
    metadata["signer_id"].nunique()
)


print("\nFichier créé :")
print(OUTPUT_CSV)


print("\nPremières lignes :")

print(
    metadata.head(20).to_string(index=False)
)