from pathlib import Path
import argparse
import subprocess
import time

import pandas as pd
import requests


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "direct_mp4.csv"
)

OUTPUT_DIR = (
    Path.home()
    / "WLASL"
    / "start_kit"
    / "candidate_batch1_raw"
    / "direct_mp4"
)

REPORT = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
    / "direct_mp4_download_report.csv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Nombre maximum de vidéos à traiter"
)

args = parser.parse_args()


# ============================================================
# VALIDATION VIDEO AVEC FFPROBE
# ============================================================

def validate_video(path):

    command = [
        "ffprobe",
        "-v",
        "error",

        "-select_streams",
        "v:0",

        "-show_entries",
        "stream=codec_name,width,height,duration",

        "-of",
        "default=noprint_wrappers=1",

        str(path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        return (
            False,
            result.stderr.strip()[-300:]
        )

    if not result.stdout.strip():

        return (
            False,
            "no_video_stream"
        )

    return (
        True,
        ""
    )


# ============================================================
# CHARGER LES DONNEES
# ============================================================

df = pd.read_csv(
    INPUT,
    dtype={"video": str}
)

if args.limit is not None:

    df = df.head(
        args.limit
    )


print("=" * 80)
print("TÉLÉCHARGEMENT DIRECT_MP4")
print("=" * 80)

print(
    "Vidéos à traiter :",
    len(df)
)


# ============================================================
# SESSION HTTP
# ============================================================

session = requests.Session()

session.headers.update({

    "User-Agent":
        "Mozilla/5.0 "
        "(X11; Linux x86_64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/128 Safari/537.36"

})


rows = []


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

for _, row in df.iterrows():

    video = row["video"]
    url = row["url"]
    gloss = row["gloss"]
    source = row["source"]

    output = (
        OUTPUT_DIR
        / video
    )


    result_row = {

        "video":
            video,

        "gloss":
            gloss,

        "source":
            source,

        "url":
            url,

        "http_status":
            None,

        "content_type":
            None,

        "content_length_header":
            None,

        "downloaded_bytes":
            0,

        "ffprobe_ok":
            False,

        "success":
            False,

        "status":
            "",

        "error":
            ""
    }


    print(
        f"\n[{len(rows) + 1}/{len(df)}]",
        video,
        gloss,
        "|",
        source
    )


    # ========================================================
    # 1. FICHIER DEJA PRESENT
    # ========================================================

    if (
        output.exists()
        and output.stat().st_size > 1024
    ):

        ok, error = validate_video(
            output
        )

        result_row[
            "downloaded_bytes"
        ] = output.stat().st_size

        result_row[
            "ffprobe_ok"
        ] = ok


        if ok:

            result_row[
                "success"
            ] = True

            result_row[
                "status"
            ] = "ALREADY_VALID"

            print(
                "  -> déjà présent et valide"
            )

            rows.append(
                result_row
            )

            continue


    # ========================================================
    # 2. SOURCE CONNUE INDISPONIBLE
    #
    # aslsearch renvoie actuellement une page HTML
    # au lieu d'une vraie vidéo.
    # ========================================================

    if source == "aslsearch":

        result_row[
            "status"
        ] = "KNOWN_UNAVAILABLE"

        result_row[
            "error"
        ] = (
            "aslsearch retourne une page HTML"
        )

        print(
            "  -> source connue indisponible : aslsearch"
        )

        rows.append(
            result_row
        )

        continue


    # ========================================================
    # 3. FICHIER TEMPORAIRE
    # ========================================================

    temp = output.with_suffix(
        ".download"
    )


    try:

        # ====================================================
        # REQUETE HTTP
        # ====================================================

        response = session.get(
            url,
            timeout=30,
            stream=True,
            allow_redirects=True
        )


        result_row[
            "http_status"
        ] = response.status_code


        content_type = (
            response.headers
            .get(
                "Content-Type",
                ""
            )
            .lower()
        )


        result_row[
            "content_type"
        ] = content_type


        result_row[
            "content_length_header"
        ] = (
            response.headers
            .get(
                "Content-Length"
            )
        )


        # ====================================================
        # HTTP != 200
        # ====================================================

        if response.status_code != 200:

            result_row[
                "status"
            ] = "HTTP_ERROR"

            result_row[
                "error"
            ] = (
                f"HTTP {response.status_code}"
            )

            print(
                "  -> erreur HTTP",
                response.status_code
            )

            rows.append(
                result_row
            )

            continue


        # ====================================================
        # BLOQUER LES PAGES HTML
        # ====================================================

        if (
            "text/html"
            in content_type
        ):

            result_row[
                "status"
            ] = "HTML_NOT_VIDEO"

            result_row[
                "error"
            ] = (
                "server returned text/html"
            )

            print(
                "  -> HTML, pas une vidéo"
            )

            rows.append(
                result_row
            )

            continue


        # ====================================================
        # TELECHARGEMENT
        # ====================================================

        total = 0


        with open(
            temp,
            "wb"
        ) as f:

            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):

                if not chunk:
                    continue

                f.write(
                    chunk
                )

                total += len(
                    chunk
                )


        result_row[
            "downloaded_bytes"
        ] = total


        # ====================================================
        # TAILLE MINIMALE
        # ====================================================

        if total < 1024:

            result_row[
                "status"
            ] = "TOO_SMALL"

            result_row[
                "error"
            ] = (
                f"only {total} bytes"
            )

            temp.unlink(
                missing_ok=True
            )

            print(
                "  -> fichier trop petit :",
                total,
                "octets"
            )

            rows.append(
                result_row
            )

            continue


        # ====================================================
        # VALIDATION FFPROBE
        # ====================================================

        ok, error = validate_video(
            temp
        )


        result_row[
            "ffprobe_ok"
        ] = ok


        if not ok:

            result_row[
                "status"
            ] = "INVALID_VIDEO"

            result_row[
                "error"
            ] = error

            temp.unlink(
                missing_ok=True
            )

            print(
                "  -> fichier invalide"
            )

            rows.append(
                result_row
            )

            continue


        # ====================================================
        # VIDEO VALIDEE
        # ====================================================

        temp.replace(
            output
        )


        result_row[
            "success"
        ] = True

        result_row[
            "status"
        ] = "DOWNLOADED_VALID"


        print(
            "  -> OK",
            round(
                total
                / 1024
                / 1024,
                2
            ),
            "MB"
        )


    except Exception as e:

        temp.unlink(
            missing_ok=True
        )

        result_row[
            "status"
        ] = "EXCEPTION"

        result_row[
            "error"
        ] = str(e)[:500]

        print(
            "  -> exception :",
            e
        )


    rows.append(
        result_row
    )


    # petite pause entre les requêtes
    time.sleep(
        0.3
    )


# ============================================================
# RAPPORT DE CETTE EXECUTION
# ============================================================

new_report = pd.DataFrame(
    rows
)


# ============================================================
# FUSIONNER AVEC ANCIEN RAPPORT
# ============================================================

if REPORT.exists():

    old = pd.read_csv(
        REPORT
    )

    old = old[
        ~old["video"].isin(
            new_report["video"]
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
# RESULTATS DE CETTE EXECUTION
# ============================================================

print("\n")
print("=" * 80)
print("RÉSULTAT DIRECT_MP4")
print("=" * 80)


print(
    "Tentatives :",
    len(new_report)
)


print(
    "Succès :",
    int(
        new_report[
            "success"
        ].sum()
    )
)


print(
    "Échecs :",
    int(
        (
            ~new_report[
                "success"
            ]
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


print(
    "\nPar source :"
)

source_report = (
    new_report
    .groupby(
        "source",
        dropna=False
    )
    .agg(
        total=(
            "video",
            "count"
        ),

        success=(
            "success",
            "sum"
        )
    )
)

source_report[
    "failed"
] = (
    source_report[
        "total"
    ]
    - source_report[
        "success"
    ]
)


print(
    source_report
    .to_string()
)


# ============================================================
# RAPPORT GLOBAL CUMULE
# ============================================================

print("\n")
print("=" * 80)
print("RAPPORT CUMULÉ")
print("=" * 80)


print(
    "Lignes dans le rapport :",
    len(final_report)
)


print(
    "Vidéos valides cumulées :",
    int(
        final_report[
            "success"
        ].sum()
    )
)


print(
    "\nStatuts cumulés :"
)

print(
    final_report[
        "status"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    "\nDossier vidéos :"
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
