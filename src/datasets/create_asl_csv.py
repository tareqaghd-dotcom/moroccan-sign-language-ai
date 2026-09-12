import json
import pandas as pd

JSON_PATH = "data/asl/WLASL100.json"
CSV_PATH = "data/asl/asl_raw.csv"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for entry in data:

    gloss = entry["gloss"].upper()

    for instance in entry["instances"]:

        video_id = instance["video_id"]
        split = instance["split"]

        rows.append({
            "video": video_id + ".mp4",
            "gloss": gloss,
            "split": split
        })

df = pd.DataFrame(rows)

df.to_csv(CSV_PATH, index=False)

print("CSV créé :", CSV_PATH)
print("Nombre de vidéos attendues :", len(df))
print("Nombre de classes :", df["gloss"].nunique())

print("\nRépartition :")
print(df["split"].value_counts())
