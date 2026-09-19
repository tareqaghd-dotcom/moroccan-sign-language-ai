from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = ROOT / "data" / "asl" / "failed_videos_metadata.csv"
OUTPUT_CSV = ROOT / "data" / "asl" / "failed_urls_status.csv"


df = pd.read_csv(
    INPUT_CSV,
    dtype={"video_id": str}
)


print("Nombre d'URLs à vérifier :", len(df))


results = []


for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Vérification des URLs"
):

    url = row["url"]

    result = {
        "video_id": row["video_id"],
        "video": row["video"],
        "gloss": row["gloss"],
        "url": url,
        "source": row["source"],
        "signer_id": row["signer_id"],
        "split": row["split"],

        "accessible": False,
        "status_code": None,
        "content_type": None,
        "content_length": None,
        "final_url": None,
        "error": ""
    }


    try:

        # On utilise GET en mode stream car certains serveurs
        # ne répondent pas correctement aux requêtes HEAD.
        response = requests.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=15
        )

        result["status_code"] = response.status_code

        result["content_type"] = response.headers.get(
            "Content-Type"
        )

        result["content_length"] = response.headers.get(
            "Content-Length"
        )

        result["final_url"] = response.url


        # URL considérée accessible si HTTP 200
        if response.status_code == 200:
            result["accessible"] = True


        response.close()


    except requests.RequestException as error:

        result["error"] = str(error)


    results.append(result)


status = pd.DataFrame(results)


status.to_csv(
    OUTPUT_CSV,
    index=False
)


print("\n========== RÉSULTAT ==========")

print("URLs analysées :", len(status))

print(
    "URLs accessibles :",
    int(status["accessible"].sum())
)

print(
    "URLs non accessibles :",
    int((~status["accessible"]).sum())
)


print("\nCodes HTTP :")

print(
    status["status_code"].value_counts(
        dropna=False
    )
)


print("\nSources :")

print(
    status["source"].value_counts(
        dropna=False
    )
)


print("\nSignants concernés :")

print(
    status["signer_id"].value_counts(
        dropna=False
    )
)


print("\nFichier créé :")

print(OUTPUT_CSV)