import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

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
        # PARAMETRES
        # ====================================================

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes


        # ====================================================
        # LSTM
        #
        # Entrée :
        # (batch, 60, 225)
        #
        # Sortie :
        # une représentation temporelle
        # ====================================================

        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout
            if num_layers > 1
            else 0.0
        )


        # ====================================================
        # DROPOUT
        # ====================================================

        self.dropout = nn.Dropout(
            dropout
        )


        # ====================================================
        # CLASSIFICATION
        #
        # 128 caractéristiques
        #          ↓
        #       100 classes
        # ====================================================

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # x :
        # (batch_size, 60, 225)

        output, (h_n, c_n) = self.lstm(
            x
        )


        # h_n :
        #
        # (num_layers, batch_size, hidden_size)
        #
        # Comme nous voulons la dernière couche :
        #
        # h_n[-1]
        #
        # donne :
        #
        # (batch_size, hidden_size)

        last_hidden = h_n[-1]


        last_hidden = self.dropout(
            last_hidden
        )


        # logits :
        #
        # (batch_size, 100)

        logits = self.classifier(
            last_hidden
        )


        return logits


# ============================================================
# TEST RAPIDE
# ============================================================

if __name__ == "__main__":

    model = LSTMClassifier()


    # Faux batch :
    #
    # 32 vidéos
    # 60 frames
    # 225 features

    x = torch.randn(
        32,
        60,
        225
    )


    output = model(
        x
    )


    print(
        "Entrée :",
        x.shape
    )

    print(
        "Sortie :",
        output.shape
    )


    print(
        "\nNombre de paramètres :",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )
