from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# RACINE DU PROJET
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_FILE = (
    ROOT
    / "data"
    / "asl"
    / "final_dataset_manifest.csv"
)


# ============================================================
# DATASET
# ============================================================

class ASLDataset(Dataset):

    def __init__(
        self,
        split,
        augment=False
    ):

        if split not in ["train", "val", "test"]:
            raise ValueError(
                "split doit être 'train', 'val' ou 'test'"
            )

        self.split = split

        # Augmentation autorisée uniquement sur train
        self.augment = (
            augment
            and split == "train"
        )

        df = pd.read_csv(
            MANIFEST_FILE
        )

        self.df = (
            df[
                df["split"] == split
            ]
            .reset_index(drop=True)
        )

        print(
            f"ASLDataset [{split}] : "
            f"{len(self.df)} exemples "
            f"| augmentation={self.augment}"
        )


    # ========================================================
    # NOMBRE D'EXEMPLES
    # ========================================================

    def __len__(self):

        return len(self.df)


    # ========================================================
    # AUGMENTATION
    # ========================================================

    def apply_augmentation(
        self,
        sequence
    ):

        # ----------------------------------------------------
        # Petit bruit gaussien
        #
        # On ne modifie que les valeurs non nulles.
        # Les zéros correspondent souvent à des landmarks
        # absents et doivent rester à zéro.
        # ----------------------------------------------------

        noise = np.random.normal(
            loc=0.0,
            scale=0.01,
            size=sequence.shape
        ).astype(np.float32)

        mask = (
            sequence != 0
        )

        augmented = sequence.copy()

        augmented[mask] += noise[mask]

        return augmented


    # ========================================================
    # CHARGER UN EXEMPLE
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]


        sequence_path = (
            ROOT
            / row["sequence_file"]
        )


        sequence = np.load(
            sequence_path
        ).astype(
            np.float32
        )


        if sequence.shape != (
            60,
            225
        ):

            raise ValueError(
                f"Shape invalide pour "
                f"{sequence_path}: "
                f"{sequence.shape}"
            )


        # ----------------------------------------------------
        # AUGMENTATION
        # ----------------------------------------------------

        if self.augment:

            sequence = (
                self.apply_augmentation(
                    sequence
                )
            )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = int(
            row["label_id"]
        )


        # ----------------------------------------------------
        # TENSORS PYTORCH
        # ----------------------------------------------------

        x = torch.from_numpy(
            sequence
        )

        y = torch.tensor(
            label,
            dtype=torch.long
        )


        return x, y


# ============================================================
# TEST RAPIDE
# ============================================================

if __name__ == "__main__":

    train_dataset = ASLDataset(
        split="train",
        augment=True
    )

    val_dataset = ASLDataset(
        split="val",
        augment=False
    )

    x_train, y_train = (
        train_dataset[0]
    )

    x_val, y_val = (
        val_dataset[0]
    )


    print()
    print(
        "Train shape :",
        x_train.shape
    )

    print(
        "Val shape :",
        x_val.shape
    )

    print(
        "Train label :",
        y_train
    )

    print(
        "Val label :",
        y_val
    )
