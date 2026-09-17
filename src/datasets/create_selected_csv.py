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
    filename = f"{video_id}.mp4"
    path = os.path.join(VIDEOS_DIR, filename)

    return os.path.isfile(path)