# j'ai telecharger 1137 fichiers .mp4 , la prochaine étape c'est verfier les id_video qui ne sont pas asl_raw
#alors on ajouter nouveau asl.csv qui contient les id_video qui sont dans asl_raw et télecharger , puis on ajouter des video dans 
# ce nouveau asl.csv les video qui on peux les telecharger

import os
import pandas as pd

RAW_CSV = "data/asl/asl_raw.csv"
SELECTED_CSV = "data/asl/asl_selected.csv"
#le chemein des video telecharger
VIDEOS_DIR = os.path.expanduser("~/WLASL/start_kit/videos")


df = pd.read_csv(RAW_CSV)


#verifier est ce que ce video existe dans /videos
def video_exists(video_id):
    filename = str(video_id)
    path = os.path.join(VIDEOS_DIR, filename)

    return os.path.isfile(path)
#
df["available"] = df["video"].apply(video_exists)
# Garde seulement les lignes avec available = True
selected = df[df["available"]].copy()
# supprimer colone available car on est pas besoin
selected = selected.drop(columns=["available"])
#sauvergarder sous forme csv dans le chemain SELECTED_CSV
selected.to_csv(SELECTED_CSV, index=False)


print("Instances dans asl_raw.csv :", len(df))
print("Vidéos disponibles :", len(selected))
print("Vidéos manquantes :", len(df) - len(selected))

print("Nombre de classes :", selected["gloss"].nunique())

print("\nRépartition train / val / test :")
print(selected["split"].value_counts())
