from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.asl_dataset import ASLDataset
from src.models.lstm_classifier_v4 import LSTMClassifierV4


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "lstm_v6_best.pt"
)


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
# DATASET
# ============================================================

val_dataset = ASLDataset("val")

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

config = checkpoint["config"]


model = LSTMClassifierV4(
    input_size=config["input_size"],
    hidden_size=config["hidden_size"],
    num_layers=config["num_layers"],
    num_classes=config["num_classes"],
    dropout=config["dropout"]
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ============================================================
# PREDICTIONS
# ============================================================

true_labels = []
pred_labels = []


with torch.no_grad():

    for x, y in val_loader:

        x = x.to(device)

        logits = model(x)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        true_labels.extend(
            y.tolist()
        )

        pred_labels.extend(
            predictions.cpu().tolist()
        )


# ============================================================
# ANALYSE
# ============================================================

prediction_counts = Counter(
    pred_labels
)


print()
print("===== ANALYSE DES PREDICTIONS =====")

print(
    "Nombre total de classes :",
    100
)

print(
    "Nombre de classes réellement prédites :",
    len(prediction_counts)
)


print()
print("10 labels les plus souvent prédits :")

for label, count in prediction_counts.most_common(10):

    print(
        f"Label {label:3d} : "
        f"{count} prédictions"
    )


# ============================================================
# CONFUSIONS
# ============================================================

confusions = Counter()

for true, pred in zip(
    true_labels,
    pred_labels
):

    if true != pred:

        confusions[
            (true, pred)
        ] += 1


print()
print("===== CONFUSIONS LES PLUS FREQUENTES =====")

for (true, pred), count in confusions.most_common(15):

    print(
        f"Vrai {true:3d} "
        f"-> prédit {pred:3d} "
        f": {count} fois"
    )
