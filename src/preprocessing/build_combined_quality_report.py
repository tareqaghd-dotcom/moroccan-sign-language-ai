from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TECHNICAL = (
    ROOT
    / "data"
    / "asl"
    / "asl_clean_technical.csv"
)

VISUAL = (
    ROOT
    / "data"
    / "asl"
    / "videos_to_review.csv"
)

MEDIAPIPE = (
    ROOT
    / "data"
    / "asl"
    / "mediapipe_quality_report.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "combined_quality_report.csv"
)

PRIORITY_OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "priority_quality_review.csv"
)


# ============================================================
# 1. CHARGEMENT
# ============================================================

technical = pd.read_csv(TECHNICAL)
visual = pd.read_csv(VISUAL)
mediapipe = pd.read_csv(MEDIAPIPE)


print("=" * 70)
print("CONSTRUCTION DU RAPPORT COMBINÉ")
print("=" * 70)

print("Vidéos techniques :", len(technical))
print("Vidéos avec warning visuel :", len(visual))
print("Vidéos MediaPipe :", len(mediapipe))


# ============================================================
# 2. WARNINGS VISUELS
# ============================================================

warning_columns = [
    "warning_decode",
    "warning_black",
    "warning_frozen",
    "warning_blur",
    "warning_low_contrast",
    "warning_low_motion",
]


# Garder seulement les colonnes existantes
warning_columns = [
    col
    for col in warning_columns
    if col in visual.columns
]


visual_small = visual[
    ["video"] + warning_columns
].copy()


# ============================================================
# 3. MERGE TECHNIQUE + VISUEL
# ============================================================

combined = technical.merge(
    visual_small,
    on="video",
    how="left",
    suffixes=("", "_visual")
)


# Vidéo absente de videos_to_review
# => aucun warning visuel
for col in warning_columns:

    combined[col] = (
        combined[col]
        .fillna(False)
        .astype(bool)
    )


combined["visual_warning_count"] = (
    combined[warning_columns]
    .sum(axis=1)
)


combined["has_visual_warning"] = (
    combined["visual_warning_count"] > 0
)


# ============================================================
# 4. AJOUT MEDIAPIPE
# ============================================================

# Pour éviter de dupliquer gloss/split
mp_columns = [
    col
    for col in mediapipe.columns
    if col not in ["gloss", "split"]
]


combined = combined.merge(
    mediapipe[mp_columns],
    on="video",
    how="left"
)


# ============================================================
# 5. INDICATEURS IA
# ============================================================

combined["warning_no_hand"] = (
    combined["any_hand_ratio"] == 0
)

combined["warning_hand_lt_25"] = (
    combined["any_hand_ratio"] < 0.25
)

combined["warning_hand_lt_50"] = (
    combined["any_hand_ratio"] < 0.50
)

combined["warning_pose_lt_50"] = (
    combined["pose_ratio"] < 0.50
)

combined["warning_no_hand_middle"] = (
    combined["middle_hand_ratio"] == 0
)

combined["warning_long_no_hand"] = (
    combined["longest_no_hand_run_ratio"] > 0.75
)


ai_warning_columns = [
    "warning_no_hand",
    "warning_hand_lt_25",
    "warning_pose_lt_50",
    "warning_no_hand_middle",
    "warning_long_no_hand",
]


combined["ai_warning_count"] = (
    combined[ai_warning_columns]
    .sum(axis=1)
)


# ============================================================
# 6. PRIORITÉ DE REVUE
#
# Ce n'est PAS un score de qualité.
# C'est uniquement le nombre d'indicateurs suspects.
# ============================================================

combined["review_indicator_count"] = (
    combined["visual_warning_count"]
    + combined["ai_warning_count"]
)


combined["needs_review"] = (
    combined["review_indicator_count"] > 0
)


# ============================================================
# 7. SAUVEGARDE
# ============================================================

combined.to_csv(
    OUTPUT,
    index=False
)


priority = (
    combined[
        combined["needs_review"]
    ]
    .sort_values(
        [
            "review_indicator_count",
            "any_hand_ratio",
            "pose_ratio"
        ],
        ascending=[
            False,
            True,
            True
        ]
    )
)


priority.to_csv(
    PRIORITY_OUTPUT,
    index=False
)


# ============================================================
# 8. STATISTIQUES
# ============================================================

print("\n")
print("=" * 70)
print("RÉSULTAT DU RAPPORT COMBINÉ")
print("=" * 70)

print(
    "Vidéos analysées :",
    len(combined)
)

print(
    "Warnings visuels :",
    int(
        combined[
            "has_visual_warning"
        ].sum()
    )
)

print(
    "\nMediaPipe any_hand = 0 :",
    int(
        combined[
            "warning_no_hand"
        ].sum()
    )
)

print(
    "MediaPipe any_hand < 25 % :",
    int(
        combined[
            "warning_hand_lt_25"
        ].sum()
    )
)

print(
    "MediaPipe any_hand < 50 % :",
    int(
        combined[
            "warning_hand_lt_50"
        ].sum()
    )
)

print(
    "Pose < 50 % :",
    int(
        combined[
            "warning_pose_lt_50"
        ].sum()
    )
)

print(
    "Aucune main dans les 20-80 % centraux :",
    int(
        combined[
            "warning_no_hand_middle"
        ].sum()
    )
)

print(
    "Plus de 75 % consécutifs sans main :",
    int(
        combined[
            "warning_long_no_hand"
        ].sum()
    )
)


print(
    "\nWarning visuel + main < 25 % :",
    int(
        (
            combined[
                "has_visual_warning"
            ]
            &
            combined[
                "warning_hand_lt_25"
            ]
        ).sum()
    )
)


print(
    "\nVidéos avec au moins un indicateur :",
    int(
        combined[
            "needs_review"
        ].sum()
    )
)


print(
    "\nRépartition du nombre d'indicateurs :"
)

print(
    combined[
        "review_indicator_count"
    ]
    .value_counts()
    .sort_index()
)


print("\n========== 20 VIDÉOS À REGARDER EN PREMIER ==========")

columns_to_show = [
    "video",
    "gloss",
    "split",
    "visual_warning_count",
    "any_hand_ratio",
    "middle_hand_ratio",
    "pose_ratio",
    "longest_no_hand_run_ratio",
    "review_indicator_count",
]

columns_to_show = [
    col
    for col in columns_to_show
    if col in priority.columns
]

print(
    priority[
        columns_to_show
    ]
    .head(20)
    .to_string(index=False)
)


print("\nRapport complet :")
print(OUTPUT)

print("\nVidéos prioritaires :")
print(PRIORITY_OUTPUT)