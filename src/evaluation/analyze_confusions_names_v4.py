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
    / "lstm_v4_best.pt"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "asl"
    / "final_dataset_manifest.csv"
)


# ============================================================
# LABEL -> NOM DU SIGNE
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
# VALIDATION
# ============================================================

val_dataset = ASLDataset(
    "val",
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# ============================================================
# MODELE V3
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
    "===== CONFUSIONS V4 AVEC NOMS ====="
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
