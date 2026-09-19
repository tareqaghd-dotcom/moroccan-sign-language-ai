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

CANDIDATES_CSV = (
    ROOT
    / "data"
    / "asl"
    / "full_wlasl_class_candidates.csv"
)

CURRENT_REPORT = (
    ROOT
    / "data"
    / "asl"
    / "class_quality_report.csv"
)

OUTPUT_JSON = (
    ROOT
    / "data"
    / "asl"
    / "wlasl_candidate_pool.json"
)

OUTPUT_SUMMARY = (
    ROOT
    / "data"
    / "asl"
    / "wlasl_candidate_pool_summary.csv"
)


# ============================================================
# PARAMÈTRE
# ============================================================

N_NEW_CLASSES = 150


# ============================================================
# 1. CHARGER LES CLASSES ACTUELLES
# ============================================================

current = pd.read_csv(
    CURRENT_REPORT
)

current_glosses = set(
    current["gloss"]
    .str.lower()
)


print("=" * 75)
print("CRÉATION DU POOL DE CLASSES")
print("=" * 75)

print(
    "Classes actuelles :",
    len(current_glosses)
)


# ============================================================
# 2. CHARGER LES CANDIDATES
# ============================================================

candidates = pd.read_csv(
    CANDIDATES_CSV
)

candidates["gloss_lower"] = (
    candidates["gloss"]
    .str.lower()
)


# Enlever les 100 classes déjà présentes
new_candidates = candidates[
    ~candidates[
        "gloss_lower"
    ].isin(current_glosses)
].copy()


# Le fichier est déjà trié selon :
# total_instances
# n_signers
# n_sources
# dominance
selected_new = (
    new_candidates
    .head(N_NEW_CLASSES)
    .copy()
)


new_glosses = set(
    selected_new[
        "gloss_lower"
    ]
)


print(
    "Nouvelles classes sélectionnées :",
    len(new_glosses)
)


# ============================================================
# 3. UNION
# ============================================================

selected_glosses = (
    current_glosses
    | new_glosses
)


print(
    "Pool total :",
    len(selected_glosses)
)


if len(selected_glosses) != (
    100 + N_NEW_CLASSES
):

    print(
        "ATTENTION : nombre différent de celui attendu."
    )


# ============================================================
# 4. CHARGER WLASL COMPLET
# ============================================================

with open(
    FULL_JSON,
    "r",
    encoding="utf-8"
) as f:

    full_data = json.load(f)


# ============================================================
# 5. EXTRAIRE LES 250 CLASSES
# ============================================================

pool_json = []

summary_rows = []


for item in full_data:

    gloss = item["gloss"]

    gloss_lower = gloss.lower()


    if gloss_lower not in selected_glosses:
        continue


    pool_json.append(item)


    instances = item["instances"]


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
        pd.Series(
            signer_ids,
            dtype="object"
        )
        .value_counts()
    )


    if len(signer_counts) > 0:

        max_same_signer = int(
            signer_counts.iloc[0]
        )

        signer_dominance = (
            max_same_signer
            / len(instances)
        )

    else:

        max_same_signer = 0
        signer_dominance = 0


    category = (
        "CURRENT"
        if gloss_lower in current_glosses
        else "NEW_CANDIDATE"
    )


    summary_rows.append({

        "gloss":
            gloss,

        "category":
            category,

        "total_instances":
            len(instances),

        "n_signers":
            len(set(signer_ids)),

        "n_sources":
            len(set(sources)),

        "max_videos_same_signer":
            max_same_signer,

        "signer_dominance":
            signer_dominance
    })


# ============================================================
# 6. SAUVEGARDE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        pool_json,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 7. SAUVEGARDE RÉSUMÉ
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


summary = summary.sort_values(
    [
        "category",
        "total_instances",
        "n_signers",
        "n_sources"
    ],
    ascending=[
        True,
        False,
        False,
        False
    ]
)


summary.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# 8. STATISTIQUES
# ============================================================

print("\n")
print("=" * 75)
print("POOL FINAL")
print("=" * 75)

print(
    "Classes JSON :",
    len(pool_json)
)

print(
    "Classes CURRENT :",
    int(
        (
            summary["category"]
            == "CURRENT"
        ).sum()
    )
)

print(
    "Classes NEW_CANDIDATE :",
    int(
        (
            summary["category"]
            == "NEW_CANDIDATE"
        ).sum()
    )
)


print(
    "\nInstances théoriques totales :",
    int(
        summary[
            "total_instances"
        ].sum()
    )
)


print(
    "\nInstances par classe :"
)

print(
    summary[
        "total_instances"
    ]
    .describe()
)


print(
    "\nSigners par classe :"
)

print(
    summary[
        "n_signers"
    ]
    .describe()
)


print(
    "\n========== 30 NOUVELLES CLASSES DU POOL =========="
)

print(
    summary[
        summary["category"]
        == "NEW_CANDIDATE"
    ][
        [
            "gloss",
            "total_instances",
            "n_signers",
            "n_sources",
            "signer_dominance"
        ]
    ]
    .head(30)
    .to_string(index=False)
)


print("\nJSON créé :")
print(OUTPUT_JSON)

print("\nRésumé créé :")
print(OUTPUT_SUMMARY)