from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.asl_dataset import ASLDataset
from src.models.lstm_classifier_v4 import LSTMClassifierV4


# ============================================================
# CHEMINS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "lstm_v6_best.pt"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "asl"
    / "final_dataset_manifest.csv"
)


# ============================================================
# LABEL -> NOM
# ============================================================

manifest = pd.read_csv(
    MANIFEST_PATH
)

label_to_gloss = (
    manifest[
        ["label_id", "gloss"]
    ]
    .drop_duplicates()
    .set_index("label_id")["gloss"]
    .to_dict()
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
# TEST DATASET
#
# IMPORTANT :
# aucune augmentation sur TEST
# ============================================================

test_dataset = ASLDataset(
    "test",
    augment=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# CHECKPOINT V6
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
    "Accuracy validation du modèle choisi :",
    f'{checkpoint["val_accuracy"]:.2f}%'
)


# ============================================================
# RECREER EXACTEMENT V6
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
# TEST
# ============================================================

true_labels = []
pred_labels = []

correct = 0
total = 0


with torch.no_grad():

    for x, y in test_loader:

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

        true_labels.extend(
            y.cpu().tolist()
        )

        pred_labels.extend(
            predictions.cpu().tolist()
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
print("===== TEST FINAL V6 =====")

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
    "Accuracy TEST :",
    f"{accuracy:.2f}%"
)


# ============================================================
# NOMBRE DE CLASSES PREDITES
# ============================================================

prediction_counts = Counter(
    pred_labels
)

print()
print(
    "Classes réellement prédites :",
    f"{len(prediction_counts)}/100"
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
print(
    "===== CONFUSIONS TEST LES PLUS FREQUENTES ====="
)

for (
    true,
    pred
), count in confusions.most_common(20):

    true_name = label_to_gloss.get(
        true,
        "?"
    )

    pred_name = label_to_gloss.get(
        pred,
        "?"
    )

    print(
        f"{true:3d} {true_name:15s}"
        f" -> "
        f"{pred:3d} {pred_name:15s}"
        f" : {count} fois"
    )
