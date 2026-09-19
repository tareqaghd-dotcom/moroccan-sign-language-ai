from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "asl"
    / "combined_quality_report.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "automatic_quality_decision.csv"
)

REVIEW_OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "automatic_manual_review.csv"
)


df = pd.read_csv(INPUT)


# ============================================================
# DÉCISION PAR DÉFAUT
# ============================================================

df["automatic_decision"] = "AUTO_KEEP"
df["automatic_reason"] = "aucun problème sérieux détecté"


# ============================================================
# CAS À VÉRIFIER
#
# IMPORTANT :
# On ne rejette PAS automatiquement.
# On sélectionne uniquement les vidéos réellement suspectes.
# ============================================================

review = (

    # problème de pose réellement important
    (df["pose_ratio"] < 0.90)

    |

    # très mauvaise détection des mains
    # + autre signal suspect
    (
        (df["any_hand_ratio"] < 0.15)
        &
        (df["visual_warning_count"] >= 1)
    )

    |

    # très faible présence des mains au centre
    # + warning visuel
    (
        (df["middle_hand_ratio"] < 0.10)
        &
        (df["visual_warning_count"] >= 1)
    )

    |

    # beaucoup de warnings visuels
    (df["visual_warning_count"] >= 4)

    |

    # très longue période sans main
    (df["longest_no_hand_run_ratio"] > 0.80)
)


df.loc[
    review,
    "automatic_decision"
] = "REVIEW"


# ============================================================
# EXPLIQUER POURQUOI
# ============================================================

def reason(row):

    reasons = []

    if row["pose_ratio"] < 0.90:
        reasons.append("pose faible")

    if (
        row["any_hand_ratio"] < 0.15
        and row["visual_warning_count"] >= 1
    ):
        reasons.append(
            "main très faible + warning visuel"
        )

    if (
        row["middle_hand_ratio"] < 0.10
        and row["visual_warning_count"] >= 1
    ):
        reasons.append(
            "main très faible au centre + warning visuel"
        )

    if row["visual_warning_count"] >= 4:
        reasons.append(
            "nombreux warnings visuels"
        )

    if row["longest_no_hand_run_ratio"] > 0.80:
        reasons.append(
            "très longue séquence sans main"
        )

    if not reasons:
        return "aucun problème sérieux détecté"

    return " ; ".join(reasons)


df["automatic_reason"] = df.apply(
    reason,
    axis=1
)


# ============================================================
# SAUVEGARDE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)


manual = (
    df[
        df["automatic_decision"] == "REVIEW"
    ]
    .sort_values(
        [
            "visual_warning_count",
            "any_hand_ratio"
        ],
        ascending=[
            False,
            True
        ]
    )
)


manual.to_csv(
    REVIEW_OUTPUT,
    index=False
)


# ============================================================
# AFFICHAGE
# ============================================================

print("=" * 70)
print("DÉCISION QUALITÉ AUTOMATIQUE")
print("=" * 70)

print(
    "\nNombre total :",
    len(df)
)

print(
    "\nDécisions :"
)

print(
    df["automatic_decision"]
    .value_counts()
)


print(
    "\nVidéos à vérifier manuellement :",
    len(manual)
)


print(
    "\n========== VIDÉOS À VÉRIFIER =========="
)

cols = [
    "video",
    "gloss",
    "split",
    "any_hand_ratio",
    "middle_hand_ratio",
    "pose_ratio",
    "visual_warning_count",
    "longest_no_hand_run_ratio",
    "automatic_reason"
]

print(
    manual[cols]
    .to_string(index=False)
)


print("\nRapport complet :")
print(OUTPUT)

print("\nReview manuel :")
print(REVIEW_OUTPUT)