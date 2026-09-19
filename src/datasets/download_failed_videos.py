from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

METADATA_CSV = (
    ROOT / "data" / "asl" / "failed_videos_metadata.csv"
)

DOWNLOAD_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "recovery_downloads"
)

REPORT_CSV = (
    ROOT / "data" / "asl" / "recovery_download_report.csv"
)


DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    METADATA_CSV,
    dtype={"video_id": str}
)


results = []


for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Téléchargement des 114 sources"
):

    video_id = row["video_id"]

    # conserver les zéros initiaux
    video_id = video_id.zfill(5)

    url = row["url"]

    output = DOWNLOAD_DIR / f"{video_id}.mp4"


    result = {
        "video_id": video_id,
        "gloss": row["gloss"],
        "url": url,
        "download_ok": False,
        "status_code": None,
        "content_type": None,
        "size_bytes": 0,
        "error": ""
    }


    try:

        response = requests.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=30
        )

        result["status_code"] = response.status_code

        result["content_type"] = response.headers.get(
            "Content-Type"
        )


        if response.status_code != 200:

            result["error"] = (
                f"HTTP {response.status_code}"
            )

            response.close()

            results.append(result)

            continue


        with open(output, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if chunk:
                    f.write(chunk)


        response.close()


        size = output.stat().st_size

        result["size_bytes"] = size


        # On demande au minimum 1 KB.
        # Ce n'est PAS encore un critère final de qualité.
        # Cela sert seulement à détecter les faux fichiers minuscules.

        if size > 1024:

            result["download_ok"] = True

        else:

            result["error"] = "downloaded_file_too_small"


    except requests.RequestException as error:

        result["error"] = str(error)


    except Exception as error:

        result["error"] = str(error)


    results.append(result)


report = pd.DataFrame(results)

report.to_csv(
    REPORT_CSV,
    index=False
)


print("\n")
print("=" * 60)
print("RÉSULTAT DU RETÉLÉCHARGEMENT")
print("=" * 60)

print("Sources :", len(report))

print(
    "Téléchargements > 1 KB :",
    int(report["download_ok"].sum())
)

print(
    "Échecs / fichiers minuscules :",
    int((~report["download_ok"]).sum())
)


sizes = report.loc[
    report["download_ok"],
    "size_bytes"
]


if len(sizes) > 0:

    print(
        "\nTaille minimale :",
        round(sizes.min() / 1024, 2),
        "KB"
    )

    print(
        "Taille médiane :",
        round(sizes.median() / 1024, 2),
        "KB"
    )

    print(
        "Taille maximale :",
        round(sizes.max() / 1024, 2),
        "KB"
    )


print("\nFichier rapport :")
print(REPORT_CSV)

print("\nDossier téléchargements :")
print(DOWNLOAD_DIR)