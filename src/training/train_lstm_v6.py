from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.asl_dataset import ASLDataset
from src.models.lstm_classifier_v4 import LSTMClassifierV4


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 80

INPUT_SIZE = 225
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 100
DROPOUT = 0.5

SEED = 42

# Early stopping :
# si la validation ne s'améliore pas pendant 12 epochs,
# on arrête.
EARLY_STOPPING_PATIENCE = 12


# ============================================================
# REPRODUCTIBILITE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device :", device)


# ============================================================
# DATASETS
# ============================================================

train_dataset = ASLDataset("train", augment=True)
val_dataset = ASLDataset("val", augment=False)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Train :", len(train_dataset))
print("Val :", len(val_dataset))


# ============================================================
# MODELE
# ============================================================

model = LSTMClassifierV4(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
)

model = model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# SCHEDULER
#
# Si val_loss ne baisse plus pendant 4 epochs :
# learning rate = learning rate * 0.5
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=4,
    min_lr=1e-5
)


# ============================================================
# CHECKPOINT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "lstm_v6_best.pt"
)


# ============================================================
# SUIVI DU MEILLEUR MODELE
# ============================================================

best_val_accuracy = 0.0
best_val_loss = float("inf")

epochs_without_improvement = 0


# ============================================================
# ENTRAINEMENT
# ============================================================

for epoch in range(1, NUM_EPOCHS + 1):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0


    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()


        train_loss += (
            loss.item()
            * x.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        train_correct += (
            predictions == y
        ).sum().item()

        train_total += y.size(0)


    train_loss /= train_total

    train_accuracy = (
        100.0
        * train_correct
        / train_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits,
                y
            )

            val_loss += (
                loss.item()
                * x.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            val_correct += (
                predictions == y
            ).sum().item()

            val_total += y.size(0)


    val_loss /= val_total

    val_accuracy = (
        100.0
        * val_correct
        / val_total
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        val_loss
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # ========================================================
    # AFFICHAGE
    # ========================================================

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_accuracy:.2f}% | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_accuracy:.2f}% | "
        f"LR: {current_lr:.6f}"
    )


    # ========================================================
    # SAUVEGARDE DU MEILLEUR MODELE
    #
    # On garde ici le meilleur selon Val accuracy.
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_accuracy":
                    val_accuracy,

                "val_loss":
                    val_loss,

                "config": {
                    "input_size":
                        INPUT_SIZE,

                    "hidden_size":
                        HIDDEN_SIZE,

                    "num_layers":
                        NUM_LAYERS,

                    "num_classes":
                        NUM_CLASSES,

                    "dropout":
                        DROPOUT
                }
            },

            BEST_MODEL_PATH
        )

        print(
            "  -> Nouveau meilleur modèle sauvegardé :",
            BEST_MODEL_PATH
        )


    # ========================================================
    # EARLY STOPPING
    #
    # On surveille la Val loss.
    # ========================================================

    if val_loss < best_val_loss - 1e-4:

        best_val_loss = val_loss
        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1


    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print()
        print(
            "Early stopping : "
            f"aucune amélioration de Val loss "
            f"depuis {EARLY_STOPPING_PATIENCE} epochs."
        )

        break


# ============================================================
# FIN
# ============================================================

print()
print("Entraînement terminé.")

print(
    "Meilleure accuracy validation :",
    f"{best_val_accuracy:.2f}%"
)

print(
    "Meilleure val loss :",
    f"{best_val_loss:.4f}"
)

print(
    "Meilleur modèle :",
    BEST_MODEL_PATH
)
