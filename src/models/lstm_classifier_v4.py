import torch
import torch.nn as nn


class LSTMClassifierV4(nn.Module):

    def __init__(
        self,
        input_size=225,
        hidden_size=128,
        num_layers=2,
        num_classes=100,
        dropout=0.3
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )


    def forward(self, x):

        # x :
        # (batch, 60, 225)

        output, _ = self.lstm(x)

        # output :
        # (batch, 60, 128)

        # Moyenne des 60 instants
        pooled = output.mean(dim=1)

        # pooled :
        # (batch, 128)

        pooled = self.dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        # logits :
        # (batch, 100)

        return logits


if __name__ == "__main__":

    model = LSTMClassifierV4()

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
