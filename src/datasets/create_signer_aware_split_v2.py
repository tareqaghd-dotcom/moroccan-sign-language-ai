from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "asl"
    / "final_wlasl100.csv"
)

OLD_SPLIT = (
    ROOT
    / "data"
    / "asl"
    / "final_wlasl100_split.csv"
)

OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "final_wlasl100_split_v2.csv"
)

SIGNER_OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "final_signer_split_v2.csv"
)

CLASS_OUTPUT = (
    ROOT
    / "data"
    / "asl"
    / "final_split_class_report_v2.csv"
)


# ============================================================
# PARAMÈTRES
# ============================================================

SEED = 20260919

TRIALS_PER_CONFIG = 50000

TARGET_TRAIN = 0.70
TARGET_VAL = 0.15
TARGET_TEST = 0.15


# Plusieurs nombres de signers sont essayés.
CONFIGS = [
    (36, 7, 7),
    (35, 7, 8),
    (35, 8, 7),
    (34, 8, 8),
    (34, 7, 9),
    (34, 9, 7),
]


# ============================================================
# NORMALISER SIGNER
# ============================================================

def normalize_signer(value):

    try:
        x = float(value)

        if x.is_integer():
            return str(int(x))

        return str(x)

    except Exception:
        return str(value)


# ============================================================
# CHARGEMENT
# ============================================================

df = pd.read_csv(
    INPUT,
    dtype={"video": str}
)

df["gloss"] = (
    df["gloss"]
    .astype(str)
    .str.upper()
)

df["signer_key"] = (
    df["signer_id"]
    .apply(normalize_signer)
)


signers = sorted(
    df["signer_key"].unique()
)

classes = sorted(
    df["gloss"].unique()
)


print("=" * 90)
print("SPLIT SIGNER-AWARE V2")
print("=" * 90)

print("Vidéos :", len(df))
print("Classes :", len(classes))
print("Signers :", len(signers))


# ============================================================
# MATRICE SIGNER × CLASSE
# ============================================================

pivot = pd.crosstab(
    df["signer_key"],
    df["gloss"]
)

pivot = pivot.reindex(
    index=signers,
    columns=classes,
    fill_value=0
)

matrix = pivot.to_numpy(
    dtype=np.int32
)

class_totals = matrix.sum(
    axis=0
)

signer_video_counts = matrix.sum(
    axis=1
)

total_videos = int(
    matrix.sum()
)

n_signers = len(signers)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    train_idx,
    val_idx,
    test_idx
):

    train_counts = matrix[
        train_idx
    ].sum(axis=0)

    val_counts = matrix[
        val_idx
    ].sum(axis=0)

    test_counts = matrix[
        test_idx
    ].sum(axis=0)


    # --------------------------------------------------------
    # Classes manquantes
    # --------------------------------------------------------

    missing_train = int(
        np.sum(
            train_counts == 0
        )
    )

    missing_val = int(
        np.sum(
            val_counts == 0
        )
    )

    missing_test = int(
        np.sum(
            test_counts == 0
        )
    )

    missing_total = (
        missing_train
        + missing_val
        + missing_test
    )


    # --------------------------------------------------------
    # Nombre réel de vidéos
    # --------------------------------------------------------

    train_videos = int(
        signer_video_counts[
            train_idx
        ].sum()
    )

    val_videos = int(
        signer_video_counts[
            val_idx
        ].sum()
    )

    test_videos = int(
        signer_video_counts[
            test_idx
        ].sum()
    )


    train_ratio_global = (
        train_videos
        / total_videos
    )

    val_ratio_global = (
        val_videos
        / total_videos
    )

    test_ratio_global = (
        test_videos
        / total_videos
    )


    global_ratio_error = (
        abs(
            train_ratio_global
            - TARGET_TRAIN
        )
        +
        abs(
            val_ratio_global
            - TARGET_VAL
        )
        +
        abs(
            test_ratio_global
            - TARGET_TEST
        )
    )


    # --------------------------------------------------------
    # Ratios par classe
    # --------------------------------------------------------

    train_class_ratio = (
        train_counts
        / class_totals
    )

    val_class_ratio = (
        val_counts
        / class_totals
    )

    test_class_ratio = (
        test_counts
        / class_totals
    )


    class_ratio_error = float(

        np.mean(
            np.abs(
                train_class_ratio
                - TARGET_TRAIN
            )
        )

        +

        np.mean(
            np.abs(
                val_class_ratio
                - TARGET_VAL
            )
        )

        +

        np.mean(
            np.abs(
                test_class_ratio
                - TARGET_TEST
            )
        )
    )


    # --------------------------------------------------------
    # Classes avec très peu de données d'entraînement
    # --------------------------------------------------------

    train_less_3 = int(
        np.sum(
            train_counts < 3
        )
    )

    train_less_4 = int(
        np.sum(
            train_counts < 4
        )
    )


    # --------------------------------------------------------
    # Classes avec une seule vidéo en validation/test
    # --------------------------------------------------------

    thin_val = int(
        np.sum(
            val_counts == 1
        )
    )

    thin_test = int(
        np.sum(
            test_counts == 1
        )
    )


    # ========================================================
    # SCORE
    #
    # Priorité absolue :
    # 1. aucune classe manquante
    # 2. environ 70/15/15 vidéos
    # 3. assez de train par classe
    # 4. équilibre par classe
    # ========================================================

    score = (

        missing_total * 1_000_000

        +

        global_ratio_error * 10_000

        +

        train_less_3 * 100

        +

        train_less_4 * 10

        +

        class_ratio_error * 100

        +

        thin_val * 0.5

        +

        thin_test * 0.5
    )


    return {

        "score":
            score,

        "missing_total":
            missing_total,

        "missing_train":
            missing_train,

        "missing_val":
            missing_val,

        "missing_test":
            missing_test,

        "train_videos":
            train_videos,

        "val_videos":
            val_videos,

        "test_videos":
            test_videos,

        "train_ratio":
            train_ratio_global,

        "val_ratio":
            val_ratio_global,

        "test_ratio":
            test_ratio_global,

        "train_less_3":
            train_less_3,

        "train_less_4":
            train_less_4,

        "thin_val":
            thin_val,

        "thin_test":
            thin_test,

        "train_counts":
            train_counts,

        "val_counts":
            val_counts,

        "test_counts":
            test_counts
    }


# ============================================================
# RECHERCHE ALEATOIRE
# ============================================================

rng = np.random.default_rng(
    SEED
)

best = None


for config in CONFIGS:

    n_train, n_val, n_test = config


    if (
        n_train
        + n_val
        + n_test
        != n_signers
    ):

        continue


    print("\n")
    print("=" * 90)

    print(
        "Configuration signers :",
        config
    )

    print("=" * 90)


    best_config = None


    for trial in range(
        TRIALS_PER_CONFIG
    ):

        permutation = rng.permutation(
            n_signers
        )


        train_idx = permutation[
            :n_train
        ]

        val_idx = permutation[
            n_train:
            n_train + n_val
        ]

        test_idx = permutation[
            n_train + n_val:
        ]


        result = evaluate(
            train_idx,
            val_idx,
            test_idx
        )


        candidate = {
            **result,

            "train_idx":
                train_idx.copy(),

            "val_idx":
                val_idx.copy(),

            "test_idx":
                test_idx.copy(),

            "configuration":
                config
        }


        if (
            best_config is None
            or candidate["score"]
            < best_config["score"]
        ):

            best_config = candidate


        if (
            best is None
            or candidate["score"]
            < best["score"]
        ):

            best = candidate


    print(
        "Classes manquantes :",
        best_config[
            "missing_total"
        ]
    )


    print(
        "Vidéos :",
        best_config[
            "train_videos"
        ],
        "/",
        best_config[
            "val_videos"
        ],
        "/",
        best_config[
            "test_videos"
        ]
    )


    print(
        "Ratios :",
        round(
            best_config[
                "train_ratio"
            ] * 100,
            2
        ),
        "/",
        round(
            best_config[
                "val_ratio"
            ] * 100,
            2
        ),
        "/",
        round(
            best_config[
                "test_ratio"
            ] * 100,
            2
        )
    )


    print(
        "Classes train < 3 vidéos :",
        best_config[
            "train_less_3"
        ]
    )


# ============================================================
# SI AUCUN SPLIT STRICT
# ============================================================

if best is None:

    raise RuntimeError(
        "Aucun split généré."
    )


if best[
    "missing_total"
] != 0:

    raise RuntimeError(
        "Aucun split trouvé avec les 100 classes "
        "dans train/val/test."
    )


# ============================================================
# MEILLEURE REPARTITION
# ============================================================

train_signers = {
    signers[i]
    for i
    in best[
        "train_idx"
    ]
}

val_signers = {
    signers[i]
    for i
    in best[
        "val_idx"
    ]
}

test_signers = {
    signers[i]
    for i
    in best[
        "test_idx"
    ]
}


# ============================================================
# VERIFICATION LEAKAGE
# ============================================================

if (
    train_signers & val_signers
    or train_signers & test_signers
    or val_signers & test_signers
):

    raise RuntimeError(
        "Chevauchement de signers."
    )


# ============================================================
# ATTRIBUTION
# ============================================================

def assign_split(signer):

    if signer in train_signers:
        return "train"

    if signer in val_signers:
        return "val"

    if signer in test_signers:
        return "test"

    raise RuntimeError(
        f"Signer inconnu : {signer}"
    )


df["split"] = (
    df["signer_key"]
    .apply(
        assign_split
    )
)


# ============================================================
# RAPPORT SIGNERS
# ============================================================

signer_rows = []


for signer in signers:

    signer_rows.append({

        "signer_id":
            signer,

        "split":
            assign_split(
                signer
            ),

        "videos":
            int(
                (
                    df[
                        "signer_key"
                    ]
                    == signer
                ).sum()
            ),

        "classes":
            int(
                df.loc[
                    df[
                        "signer_key"
                    ]
                    == signer,
                    "gloss"
                ]
                .nunique()
            )
    })


signer_report = pd.DataFrame(
    signer_rows
)


# ============================================================
# RAPPORT CLASSES
# ============================================================

class_report = (
    df
    .pivot_table(
        index="gloss",
        columns="split",
        values="video",
        aggfunc="count",
        fill_value=0
    )
    .reset_index()
)


for col in [
    "train",
    "val",
    "test"
]:

    if col not in class_report.columns:

        class_report[col] = 0


class_report["total"] = (

    class_report["train"]
    + class_report["val"]
    + class_report["test"]
)


class_report = class_report[
    [
        "gloss",
        "total",
        "train",
        "val",
        "test"
    ]
]


# ============================================================
# SAUVEGARDE
# ============================================================

df = df.drop(
    columns=["signer_key"]
)


df.to_csv(
    OUTPUT,
    index=False
)

signer_report.to_csv(
    SIGNER_OUTPUT,
    index=False
)

class_report.to_csv(
    CLASS_OUTPUT,
    index=False
)


# ============================================================
# ANCIEN SPLIT POUR COMPARAISON
# ============================================================

print("\n")
print("=" * 90)
print("COMPARAISON")
print("=" * 90)


if OLD_SPLIT.exists():

    old = pd.read_csv(
        OLD_SPLIT
    )


    print(
        "\nAncien split :"
    )


    print(
        (
            old[
                "split"
            ]
            .value_counts(
                normalize=True
            )
            * 100
        )
        .round(2)
    )


# ============================================================
# NOUVEAU SPLIT
# ============================================================

print(
    "\nNouveau split :"
)


counts = (
    df[
        "split"
    ]
    .value_counts()
)


percentages = (
    df[
        "split"
    ]
    .value_counts(
        normalize=True
    )
    * 100
)


print(
    counts
)


print(
    "\nPourcentages :"
)


print(
    percentages.round(2)
)


print("\n")
print("=" * 90)
print("SPLIT V2 FINAL")
print("=" * 90)


print(
    "Configuration signers :",
    best[
        "configuration"
    ]
)


print(
    "\nClasses sans train :",
    int(
        (
            class_report[
                "train"
            ]
            == 0
        ).sum()
    )
)


print(
    "Classes sans val :",
    int(
        (
            class_report[
                "val"
            ]
            == 0
        ).sum()
    )
)


print(
    "Classes sans test :",
    int(
        (
            class_report[
                "test"
            ]
            == 0
        ).sum()
    )
)


print(
    "\nMinimum vidéos TRAIN par classe :",
    int(
        class_report[
            "train"
        ].min()
    )
)


print(
    "Moyenne vidéos TRAIN par classe :",
    round(
        class_report[
            "train"
        ].mean(),
        2
    )
)


print(
    "\nMinimum vidéos VAL par classe :",
    int(
        class_report[
            "val"
        ].min()
    )
)


print(
    "Minimum vidéos TEST par classe :",
    int(
        class_report[
            "test"
        ].min()
    )
)


print(
    "\nChevauchement signers train/val :",
    len(
        train_signers
        & val_signers
    )
)


print(
    "Chevauchement signers train/test :",
    len(
        train_signers
        & test_signers
    )
)


print(
    "Chevauchement signers val/test :",
    len(
        val_signers
        & test_signers
    )
)


print(
    "\n========== CLASSES AVEC LE MOINS DE TRAIN =========="
)


print(
    class_report
    .sort_values(
        [
            "train",
            "total"
        ]
    )
    .head(20)
    .to_string(
        index=False
    )
)


print(
    "\nCSV V2 :"
)

print(
    OUTPUT
)


print(
    "\nRapport signers :"
)

print(
    SIGNER_OUTPUT
)


print(
    "\nRapport classes :"
)

print(
    CLASS_OUTPUT
)
