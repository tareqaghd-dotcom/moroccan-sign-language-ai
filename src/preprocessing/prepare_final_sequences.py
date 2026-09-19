from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 60
FEATURE_DIM = 225

ROOT = Path(__file__).resolve().parents[2]

CSV_FILE = (
    ROOT
    / "data"
    / "asl"
    / "final_wlasl100_split_v2.csv"
)

RAW_DIR = (
    ROOT
    / "data"
    / "asl"
    / "keypoints_raw"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "asl"
    / "sequences_60"
)

MANIFEST_FILE = (
    ROOT
    / "data"
    / "asl"
    / "final_dataset_manifest.csv"
)

CLASSES_FILE = (
    ROOT
    / "data"
    / "asl"
    / "classes.json"
)

REPORT_FILE = (
    ROOT
    / "data"
    / "asl"
    / "final_dataset_summary.csv"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# NORMALISATION D'UNE FRAME
#
# 0:99      = pose 33 × 3
# 99:162    = main gauche 21 × 3
# 162:225   = main droite 21 × 3
#
# Centre = milieu des épaules 11 et 12
# Echelle = distance entre les épaules
# ============================================================

def normalize_frame(frame):

    frame = frame.astype(
        np.float32
    ).copy()

    pose = (
        frame[0:99]
        .reshape(33, 3)
        .copy()
    )

    left = (
        frame[99:162]
        .reshape(21, 3)
        .copy()
    )

    right = (
        frame[162:225]
        .reshape(21, 3)
        .copy()
    )


    # --------------------------------------------------------
    # Si la pose n'est pas disponible, on garde la frame telle
    # quelle plutôt que d'inventer une normalisation.
    # --------------------------------------------------------

    if not np.any(
        pose != 0
    ):

        return frame


    left_shoulder = pose[11]
    right_shoulder = pose[12]


    center = (
        left_shoulder
        + right_shoulder
    ) / 2.0


    # Distance 2D des épaules
    shoulder_distance = np.linalg.norm(
        left_shoulder[:2]
        - right_shoulder[:2]
    )


    if (
        not np.isfinite(
            shoulder_distance
        )
        or shoulder_distance < 1e-6
    ):

        return frame


    # ========================================================
    # POSE
    # ========================================================

    pose[:, 0] = (
        pose[:, 0]
        - center[0]
    ) / shoulder_distance

    pose[:, 1] = (
        pose[:, 1]
        - center[1]
    ) / shoulder_distance

    pose[:, 2] = (
        pose[:, 2]
        - center[2]
    ) / shoulder_distance


    # ========================================================
    # MAIN GAUCHE
    #
    # Si la main n'a pas été détectée, elle reste exactement
    # à zéro.
    # ========================================================

    if np.any(
        left != 0
    ):

        left[:, 0] = (
            left[:, 0]
            - center[0]
        ) / shoulder_distance

        left[:, 1] = (
            left[:, 1]
            - center[1]
        ) / shoulder_distance

        # Le z MediaPipe Hand est relatif :
        # on le met seulement à l'échelle.
        left[:, 2] = (
            left[:, 2]
            / shoulder_distance
        )


    # ========================================================
    # MAIN DROITE
    # ========================================================

    if np.any(
        right != 0
    ):

        right[:, 0] = (
            right[:, 0]
            - center[0]
        ) / shoulder_distance

        right[:, 1] = (
            right[:, 1]
            - center[1]
        ) / shoulder_distance

        right[:, 2] = (
            right[:, 2]
            / shoulder_distance
        )


    result = np.concatenate(
        [
            pose.flatten(),
            left.flatten(),
            right.flatten()
        ]
    ).astype(
        np.float32
    )


    return result


# ============================================================
# NORMALISATION DE LA VIDEO
# ============================================================

def normalize_sequence(sequence):

    normalized = np.stack(
        [
            normalize_frame(frame)
            for frame in sequence
        ]
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# PASSAGE À EXACTEMENT 60 FRAMES
#
# On utilise un échantillonnage uniforme.
#
# Exemple :
# 120 frames -> on sélectionne uniformément 60 frames
#
# 30 frames -> certaines frames sont répétées pour arriver
#              à 60, sans inventer de coordonnées artificielles.
# ============================================================

def resample_sequence(
    sequence,
    target_length=60
):

    n_frames = len(sequence)


    if n_frames == 0:

        raise ValueError(
            "Séquence vide"
        )


    if n_frames == target_length:

        return sequence.copy()


    indices = np.linspace(
        0,
        n_frames - 1,
        target_length
    )


    indices = np.rint(
        indices
    ).astype(int)


    return sequence[
        indices
    ]


# ============================================================
# CHARGEMENT DATASET
# ============================================================

df = pd.read_csv(
    CSV_FILE,
    dtype={"video": str}
)


df["gloss"] = (
    df["gloss"]
    .astype(str)
    .str.upper()
)


print("=" * 85)
print("PRÉPARATION DATASET FINAL — 60 × 225")
print("=" * 85)

print(
    "Vidéos :",
    len(df)
)

print(
    "Classes :",
    df["gloss"].nunique()
)

print(
    "\nSplits :"
)

print(
    df["split"].value_counts()
)


# ============================================================
# LABELS
# ============================================================

classes = sorted(
    df["gloss"].unique()
)


class_to_id = {
    gloss: i
    for i, gloss
    in enumerate(classes)
}


with open(
    CLASSES_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "num_classes":
                len(classes),

            "sequence_length":
                SEQUENCE_LENGTH,

            "feature_dim":
                FEATURE_DIM,

            "class_to_id":
                class_to_id,

            "id_to_class":
                {
                    str(i): gloss
                    for gloss, i
                    in class_to_id.items()
                }
        },
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# TRAITEMENT
# ============================================================

rows = []


for position, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    video = str(
        row["video"]
    )

    gloss = str(
        row["gloss"]
    )

    split = str(
        row["split"]
    )


    raw_file = (
        RAW_DIR
        / split
        / f"{Path(video).stem}.npy"
    )


    split_output_dir = (
        OUTPUT_DIR
        / split
    )

    split_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_file = (
        split_output_dir
        / f"{Path(video).stem}.npy"
    )


    print(
        f"[{position}/{len(df)}]",
        video,
        gloss,
        "|",
        split
    )


    result = {

        "video":
            video,

        "gloss":
            gloss,

        "label_id":
            class_to_id[
                gloss
            ],

        "signer_id":
            row["signer_id"],

        "split":
            split,

        "original_frames":
            0,

        "sequence_frames":
            0,

        "feature_dim":
            FEATURE_DIM,

        "success":
            False,

        "status":
            "",

        "sequence_file":
            str(
                output_file.relative_to(
                    ROOT
                )
            )
    }


    # ========================================================
    # RAW EXISTE ?
    # ========================================================

    if not raw_file.exists():

        result[
            "status"
        ] = "RAW_MISSING"

        rows.append(
            result
        )

        print(
            "  -> RAW_MISSING"
        )

        continue


    # ========================================================
    # CHARGER RAW
    # ========================================================

    try:

        sequence = np.load(
            raw_file
        )

    except Exception:

        result[
            "status"
        ] = "LOAD_FAILED"

        rows.append(
            result
        )

        print(
            "  -> LOAD_FAILED"
        )

        continue


    # ========================================================
    # VALIDATION RAW
    # ========================================================

    if (
        sequence.ndim != 2
        or sequence.shape[1]
        != FEATURE_DIM
        or sequence.shape[0]
        == 0
    ):

        result[
            "status"
        ] = "INVALID_RAW_SHAPE"

        rows.append(
            result
        )

        print(
            "  -> shape raw invalide :",
            sequence.shape
        )

        continue


    result[
        "original_frames"
    ] = sequence.shape[0]


    # ========================================================
    # NORMALISATION
    # ========================================================

    normalized = normalize_sequence(
        sequence
    )


    # ========================================================
    # 60 FRAMES
    # ========================================================

    final_sequence = resample_sequence(
        normalized,
        SEQUENCE_LENGTH
    )


    # ========================================================
    # VALIDATION FINALE
    # ========================================================

    if final_sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_DIM
    ):

        result[
            "status"
        ] = "INVALID_FINAL_SHAPE"

        rows.append(
            result
        )

        print(
            "  -> shape finale invalide :",
            final_sequence.shape
        )

        continue


    if not np.isfinite(
        final_sequence
    ).all():

        result[
            "status"
        ] = "NAN_OR_INF"

        rows.append(
            result
        )

        print(
            "  -> NaN/Inf"
        )

        continue


    # ========================================================
    # SAUVEGARDE
    # ========================================================

    np.save(
        output_file,
        final_sequence.astype(
            np.float32
        )
    )


    result[
        "sequence_frames"
    ] = SEQUENCE_LENGTH

    result[
        "success"
    ] = True

    result[
        "status"
    ] = "OK"


    rows.append(
        result
    )


# ============================================================
# MANIFEST
# ============================================================

manifest = pd.DataFrame(
    rows
)


manifest.to_csv(
    MANIFEST_FILE,
    index=False
)


# ============================================================
# RAPPORT PAR CLASSE / SPLIT
# ============================================================

valid = manifest[
    manifest[
        "success"
    ]
].copy()


summary = (
    valid
    .pivot_table(
        index="gloss",
        columns="split",
        values="video",
        aggfunc="count",
        fill_value=0
    )
    .reset_index()
)


for split in [
    "train",
    "val",
    "test"
]:

    if split not in summary.columns:

        summary[split] = 0


summary[
    "total"
] = (
    summary["train"]
    + summary["val"]
    + summary["test"]
)


summary.to_csv(
    REPORT_FILE,
    index=False
)


# ============================================================
# VERIFICATION SIGNER LEAKAGE
# ============================================================

train_signers = set(
    valid.loc[
        valid["split"] == "train",
        "signer_id"
    ]
)

val_signers = set(
    valid.loc[
        valid["split"] == "val",
        "signer_id"
    ]
)

test_signers = set(
    valid.loc[
        valid["split"] == "test",
        "signer_id"
    ]
)


train_val_overlap = (
    train_signers
    & val_signers
)

train_test_overlap = (
    train_signers
    & test_signers
)

val_test_overlap = (
    val_signers
    & test_signers
)


# ============================================================
# RESULTAT
# ============================================================

print("\n")
print("=" * 85)
print("DATASET FINAL PRÊT")
print("=" * 85)


print(
    "Vidéos traitées :",
    len(manifest)
)


print(
    "Succès :",
    int(
        manifest[
            "success"
        ].sum()
    )
)


print(
    "Échecs :",
    int(
        (
            ~manifest[
                "success"
            ]
        ).sum()
    )
)


print(
    "\nStatuts :"
)

print(
    manifest[
        "status"
    ]
    .value_counts()
)


print(
    "\nShape finale :",
    (
        SEQUENCE_LENGTH,
        FEATURE_DIM
    )
)


print(
    "\nSplits :"
)

print(
    valid[
        "split"
    ]
    .value_counts()
)


print(
    "\nClasses :",
    valid[
        "gloss"
    ].nunique()
)


print(
    "\nClasses absentes TRAIN :",
    int(
        (
            summary[
                "train"
            ]
            == 0
        ).sum()
    )
)


print(
    "Classes absentes VAL :",
    int(
        (
            summary[
                "val"
            ]
            == 0
        ).sum()
    )
)


print(
    "Classes absentes TEST :",
    int(
        (
            summary[
                "test"
            ]
            == 0
        ).sum()
    )
)


print(
    "\nChevauchement signer train/val :",
    len(
        train_val_overlap
    )
)


print(
    "Chevauchement signer train/test :",
    len(
        train_test_overlap
    )
)


print(
    "Chevauchement signer val/test :",
    len(
        val_test_overlap
    )
)


print(
    "\nDossier séquences finales :"
)

print(
    OUTPUT_DIR
)


print(
    "\nManifest :"
)

print(
    MANIFEST_FILE
)


print(
    "\nClasses :"
)

print(
    CLASSES_FILE
)


print(
    "\nRésumé :"
)

print(
    REPORT_FILE
)


if (
    manifest["success"].all()
    and valid["gloss"].nunique() == 100
    and len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):

    print("\n")
    print(
        "✅ DATASET PRÊT POUR LA PARTIE MODÈLE."
    )

else:

    print("\n")
    print(
        "⚠️ Vérifier les erreurs avant livraison."
    )
