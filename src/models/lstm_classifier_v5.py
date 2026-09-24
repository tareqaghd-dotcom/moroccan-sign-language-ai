import torch
import torch.nn as nn


class LSTMClassifierV5(nn.Module):

    def __init__(
        self,
        input_size=225,
        hidden_size=128,
        num_layers=2,
        num_classes=100,
        dropout=0.3
    ):

        super().__init__()

        # ====================================================
        # BiLSTM
        # ====================================================

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # Bidirectionnel :
        #
        # 128 forward
        # +
        # 128 backward
        # =
        # 256 valeurs
        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )


    def forward(self, x):

        # x :
        # (batch, 60, 225)

        output, _ = self.lstm(x)

        # Comme le LSTM est bidirectionnel :
        #
        # output :
        # (batch, 60, 256)

        # Moyenne temporelle sur les 60 frames
        pooled = output.mean(
            dim=1
        )

        # pooled :
        # (batch, 256)

        pooled = self.dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        # logits :
        # (batch, 100)

        return logits


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    model = LSTMClassifierV5()

    x = torch.randn(
        32,
        60,
        225
    )

    logits = model(x)

    print(
        "Entrée :",
        x.shape
    )

    print(
        "Sortie :",
        logits.shape
    )

    print(
        "Nombre de paramètres :",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )
