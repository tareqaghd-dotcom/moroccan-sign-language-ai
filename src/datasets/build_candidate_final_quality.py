from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

BATCH_DIR = (
    ROOT
    / "data"
    / "asl"
    / "candidate_batch1"
)

VIDEO_QUALITY = (
    BATCH_DIR
    / "video_quality_report.csv"
)

MEDIAPIPE = (
    BATCH_DIR
    / "mediapipe_quality_report.csv"
)

AVAILABILITY = (
    BATCH_DIR
    / "availability_after_downloads.csv"
)

CURRENT_CLASSES = (
    ROOT
    / "data"
    / "asl"
    / "class_quality_report.csv"
)

OUTPUT_COMBINED = (
    BATCH_DIR
    / "combined_quality_report.csv"
)

OUTPUT_REVIEW = (
    BATCH_DIR
    / "priority_manual_review.csv"
)

OUTPUT_CLASSES = (
    BATCH_DIR
    / "candidate_class_quality_report.csv"
)


# ============================================================
# 1. CHARGEMENT
# ============================================================

quality = pd.read_csv(
    VIDEO_QUALITY,
    dtype={"video": str}
)

mediapipe = pd.read_csv(
    MEDIAPIPE,
    dtype={"video": str}
)

availability = pd.read_csv(
    AVAILABILITY,
    dtype={"video": str}
)


# ============================================================
# 2. FUSION
# ============================================================

combined = quality.merge(
    mediapipe[
        [
            "video",
            "pose_ratio",
            "left_hand_ratio",
            "right_hand_ratio",
            "any_hand_ratio",
            "both_hands_ratio",
            "middle_hand_ratio",
            "longest_no_hand_run",
            "longest_no_hand_run_ratio",
            "hand_detection_changes",
            "left_wrist_visibility",
            "right_wrist_visibility",
            "left_elbow_visibility",
            "right_elbow_visibility",
            "left_shoulder_visibility",
            "right_shoulder_visibility"
        ]
    ],
    on="video",
    how="left"
)


combined = combined.merge(
    availability[
        [
            "video",
            "signer_id",
            "source",
            "split",
            "access_type"
        ]
    ],
    on="video",
    how="left",
    suffixes=("", "_availability")
)


# ============================================================
# 3. DÉCISION CONSERVATRICE
#
# IMPORTANT :
# aucune suppression automatique.
#
# REVIEW signifie seulement :
# "cette vidéo mérite éventuellement une vérification".
# ============================================================

def decision(row):

    reasons = []

    pose = row["pose_ratio"]
    hand = row["any_hand_ratio"]
    middle = row["middle_hand_ratio"]
    longest = row[
        "longest_no_hand_run_ratio"
    ]

    visual_count = row[
        "visual_warning_count"
    ]


    if pose < 0.90:

        reasons.append(
            "pose<0.90"
        )


    if hand < 0.15:

        reasons.append(
            "any_hand<0.15"
        )


    if middle < 0.10:

        reasons.append(
            "middle_hand<0.10"
        )


    if longest > 0.80:

        reasons.append(
            "long_no_hand>0.80"
        )


    if visual_count >= 4:

        reasons.append(
            "visual_warnings>=4"
        )


    if reasons:

        return pd.Series(
            [
                "REVIEW",
                ";".join(reasons)
            ]
        )


    return pd.Series(
        [
            "AUTO_KEEP",
            ""
        ]
    )


combined[
    [
        "quality_decision",
        "review_reason"
    ]
] = combined.apply(
    decision,
    axis=1
)


# ============================================================
# 4. RAPPORT DES VIDÉOS À REVOIR
# ============================================================

review = (
    combined[
        combined[
            "quality_decision"
        ]
        == "REVIEW"
    ]
    .sort_values(
        [
            "any_hand_ratio",
            "middle_hand_ratio"
        ]
    )
)


# ============================================================
# 5. RAPPORT PAR CLASSE
# ============================================================

class_rows = []


for gloss, group in combined.groupby(
    "gloss"
):

    keep = group[
        group[
            "quality_decision"
        ]
        == "AUTO_KEEP"
    ]


    total = len(group)

    n_keep = len(keep)

    n_review = (
        total
        - n_keep
    )


    n_signers = (
        keep[
            "signer_id"
        ]
        .nunique()
    )


    n_sources = (
        keep[
            "source"
        ]
        .nunique()
    )


    if n_keep > 0:

        signer_counts = (
            keep[
                "signer_id"
            ]
            .value_counts()
        )


        max_same_signer = int(
            signer_counts.iloc[0]
        )


        dominance = (
            max_same_signer
            / n_keep
        )

    else:

        max_same_signer = 0

        dominance = 0


    class_rows.append(
        {
            "gloss":
                gloss,

            "videos_available":
                total,

            "auto_keep":
                n_keep,

            "review":
                n_review,

            "n_signers_keep":
                n_signers,

            "n_sources_keep":
                n_sources,

            "max_same_signer":
                max_same_signer,

            "signer_dominance":
                dominance,

            "usable_ratio":
                (
                    n_keep / total
                    if total > 0
                    else 0
                )
        }
    )


classes = pd.DataFrame(
    class_rows
)


# ============================================================
# 6. FLAGS POUR LES CLASSES LES PLUS INTÉRESSANTES
#
# Ce ne sont PAS encore les 100 classes finales.
# ============================================================

classes[
    "enough_videos"
] = (
    classes[
        "auto_keep"
    ]
    >= 8
)


classes[
    "good_signer_diversity"
] = (
    classes[
        "n_signers_keep"
    ]
    >= 6
)


classes[
    "low_signer_dominance"
] = (
    classes[
        "signer_dominance"
    ]
    <= 0.50
)


classes[
    "strong_candidate"
] = (
    classes[
        "enough_videos"
    ]
    &
    classes[
        "good_signer_diversity"
    ]
    &
    classes[
        "low_signer_dominance"
    ]
)


# ============================================================
# 7. TRI
# ============================================================

classes = classes.sort_values(
    [
        "auto_keep",
        "n_signers_keep",
        "n_sources_keep",
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
# 8. SAUVEGARDE
# ============================================================

combined.to_csv(
    OUTPUT_COMBINED,
    index=False
)

review.to_csv(
    OUTPUT_REVIEW,
    index=False
)

classes.to_csv(
    OUTPUT_CLASSES,
    index=False
)


# ============================================================
# 9. AFFICHAGE GLOBAL
# ============================================================

print("=" * 85)
print("DÉCISION QUALITÉ — NOUVELLES CLASSES")
print("=" * 85)


print(
    "Vidéos analysées :",
    len(combined)
)


print(
    "AUTO_KEEP :",
    int(
        (
            combined[
                "quality_decision"
            ]
            == "AUTO_KEEP"
        ).sum()
    )
)


print(
    "REVIEW :",
    int(
        (
            combined[
                "quality_decision"
            ]
            == "REVIEW"
        ).sum()
    )
)


print(
    "\n========== VIDÉOS À REVOIR =========="
)


if len(review) == 0:

    print(
        "Aucune vidéo."
    )

else:

    print(
        review[
            [
                "video",
                "gloss",
                "any_hand_ratio",
                "middle_hand_ratio",
                "pose_ratio",
                "longest_no_hand_run_ratio",
                "visual_warning_count",
                "review_reason"
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# 10. CLASSES
# ============================================================

print(
    "\n========== CLASSES =========="
)


print(
    "Classes analysées :",
    len(classes)
)


print(
    "Classes avec >= 10 AUTO_KEEP :",
    int(
        (
            classes[
                "auto_keep"
            ]
            >= 10
        ).sum()
    )
)


print(
    "Classes avec >= 8 AUTO_KEEP :",
    int(
        (
            classes[
                "auto_keep"
            ]
            >= 8
        ).sum()
    )
)


print(
    "Strong candidates :",
    int(
        classes[
            "strong_candidate"
        ].sum()
    )
)


# ============================================================
# 11. TOP NOUVELLES CLASSES
# ============================================================

print(
    "\n========== MEILLEURES NOUVELLES CLASSES =========="
)


print(
    classes[
        [
            "gloss",
            "auto_keep",
            "review",
            "n_signers_keep",
            "n_sources_keep",
            "signer_dominance",
            "strong_candidate"
        ]
    ]
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# 12. CLASSES FAIBLES ACTUELLES
# ============================================================

current = pd.read_csv(
    CURRENT_CLASSES
)


weak_current = (
    current[
        current[
            "total_clean"
        ]
        < 8
    ]
    .sort_values(
        [
            "total_clean",
            "n_signers"
        ]
    )
)


print(
    "\n========== CLASSES FAIBLES DU DATASET ACTUEL =========="
)


print(
    weak_current[
        [
            "gloss",
            "total_clean",
            "n_signers",
            "n_sources",
            "signer_dominance"
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 13. CANDIDATES POTENTIELLES
# ============================================================

strong = classes[
    classes[
        "strong_candidate"
    ]
].copy()


print(
    "\n========== CANDIDATES POTENTIELLES AU REMPLACEMENT =========="
)


print(
    strong[
        [
            "gloss",
            "auto_keep",
            "n_signers_keep",
            "n_sources_keep",
            "signer_dominance"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nRapport combiné :"
)

print(
    OUTPUT_COMBINED
)


print(
    "\nReview :"
)

print(
    OUTPUT_REVIEW
)


print(
    "\nRapport classes :"
)

print(
    OUTPUT_CLASSES
)
