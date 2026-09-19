from pathlib import Path
import json
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

FULL_JSON = (
    ROOT
    / "data"
    / "asl"
    / "WLASL_v0.3.json"
)

CURRENT_REPORT = (
    ROOT
    / "data"
    / "asl"
    / "class_quality_report.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "full_wlasl_class_candidates.csv"
)


# ============================================================
# 1. CHARGER WLASL COMPLET
# ============================================================

with open(
    FULL_JSON,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


rows = []


for item in data:

    gloss = item["gloss"]

    instances = item["instances"]

    total = len(instances)


    signer_ids = [
        inst.get("signer_id")
        for inst in instances
        if inst.get("signer_id") is not None
    ]

    sources = [
        inst.get("source")
        for inst in instances
        if inst.get("source") is not None
    ]


    signer_counts = (
        pd.Series(signer_ids)
        .value_counts()
    )


    n_signers = len(
        set(signer_ids)
    )


    n_sources = len(
        set(sources)
    )


    if len(signer_counts) > 0:

        max_same_signer = int(
            signer_counts.iloc[0]
        )

        signer_dominance = (
            max_same_signer
            / total
        )

    else:

        max_same_signer = 0
        signer_dominance = 0


    rows.append({
        "gloss": gloss,

        "total_instances": total,

        "n_signers": n_signers,

        "n_sources": n_sources,

        "max_videos_same_signer":
            max_same_signer,

        "signer_dominance":
            signer_dominance
    })


full = pd.DataFrame(rows)


# ============================================================
# 2. CLASSES ACTUELLES
# ============================================================

current = pd.read_csv(
    CURRENT_REPORT
)

current_glosses = set(
    current["gloss"]
    .str.lower()
)


full["already_in_current_100"] = (
    full["gloss"]
    .str.lower()
    .isin(current_glosses)
)


# ============================================================
# 3. FILTRE MINIMAL
#
# On ne dit PAS encore que ce sont les meilleures.
# On élimine simplement les classes très faibles.
# ============================================================

candidates = full[
    (full["total_instances"] >= 10)
    &
    (full["n_signers"] >= 5)
    &
    (full["signer_dominance"] <= 0.50)
].copy()


# ============================================================
# 4. TRI
#
# Priorité :
# - beaucoup d'instances
# - beaucoup de signers
# - beaucoup de sources
# - faible dominance
# ============================================================

candidates = candidates.sort_values(
    [
        "total_instances",
        "n_signers",
        "n_sources",
        "signer_dominance"
    ],
    ascending=[
        False,
        False,
        False,
        True
    ]
)


# ============================================================
# 5. SAUVEGARDE
# ============================================================

candidates.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# 6. AFFICHAGE
# ============================================================

print("=" * 80)
print("ANALYSE DU WLASL COMPLET")
print("=" * 80)

print(
    "Nombre total de classes WLASL :",
    len(full)
)

print(
    "Classes candidates après filtre :",
    len(candidates)
)

print(
    "\nClasses actuelles parmi les candidates :",
    int(
        candidates[
            "already_in_current_100"
        ].sum()
    )
)


new_candidates = candidates[
    ~candidates[
        "already_in_current_100"
    ]
]


print(
    "Nouvelles classes candidates :",
    len(new_candidates)
)


print(
    "\n========== 50 NOUVELLES CLASSES LES PLUS PROMETTEUSES =========="
)

print(
    new_candidates[
        [
            "gloss",
            "total_instances",
            "n_signers",
            "n_sources",
            "max_videos_same_signer",
            "signer_dominance"
        ]
    ]
    .head(50)
    .to_string(index=False)
)


print("\nRapport créé :")
print(OUTPUT)