import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.asl_dataset import ASLDataset
from src.models.lstm_classifier_v4 import LSTMClassifierV4


# ============================================================
# RACINE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "lstm_v4_best.pt"
)

CLASSES_FILE = (
    ROOT
    / "data"
    / "asl"
    / "classes.json"
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
# DATASET VALIDATION
# ============================================================

val_dataset = ASLDataset("val")

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# CHARGER LE CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

config = checkpoint["config"]

print(
    "Checkpoint epoch :",
    checkpoint["epoch"]
)

print(
    "Accuracy validation sauvegardée :",
    f'{checkpoint["val_accuracy"]:.2f}%'
)


# ============================================================
# RECREER LE MODELE
# ============================================================

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

all_predictions = []
all_labels = []

correct = 0
total = 0


with torch.no_grad():

    for x, y in val_loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == y
        ).sum().item()

        total += y.size(0)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            y.cpu().tolist()
        )


accuracy = (
    100.0
    * correct
    / total
)


# ============================================================
# RESULTAT GLOBAL
# ============================================================

print()
print("===== VALIDATION =====")

print(
    "Nombre exemples :",
    total
)

print(
    "Bonnes prédictions :",
    correct
)

print(
    "Mauvaises prédictions :",
    total - correct
)

print(
    "Accuracy :",
    f"{accuracy:.2f}%"
)


# ============================================================
# ACCURACY PAR CLASSE
# ============================================================

with open(
    CLASSES_FILE,
    encoding="utf-8"
) as f:

    classes_data = json.load(f)


# classes.json peut avoir une structure dictionnaire.
# On utilise le label présent dans le dataset pour les stats.

class_total = {}
class_correct = {}


for true_label, pred_label in zip(
    all_labels,
    all_predictions
):

    class_total[true_label] = (
        class_total.get(
            true_label,
            0
        )
        + 1
    )

    if true_label == pred_label:

        class_correct[true_label] = (
            class_correct.get(
                true_label,
                0
            )
            + 1
        )


print()
print("===== ACCURACY PAR LABEL =====")


results = []

for label in sorted(class_total):

    total_class = class_total[label]

    correct_class = class_correct.get(
        label,
        0
    )

    acc_class = (
        100.0
        * correct_class
        / total_class
    )

    results.append(
        (
            label,
            correct_class,
            total_class,
            acc_class
        )
    )


# meilleures classes
results_sorted = sorted(
    results,
    key=lambda x: x[3],
    reverse=True
)


print()
print("10 meilleurs labels :")

for (
    label,
    correct_class,
    total_class,
    acc_class
) in results_sorted[:10]:

    print(
        f"Label {label:3d} : "
        f"{correct_class}/{total_class} "
        f"= {acc_class:.2f}%"
    )


print()
print("10 labels les moins reconnus :")

for (
    label,
    correct_class,
    total_class,
    acc_class
) in results_sorted[-10:]:

    print(
        f"Label {label:3d} : "
        f"{correct_class}/{total_class} "
        f"= {acc_class:.2f}%"
    )
