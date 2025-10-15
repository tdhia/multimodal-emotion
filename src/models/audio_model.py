import torch
import torch.nn as nn

class AudioNet(nn.Module):
    """
    Réseau de neurones pour l'extraction d'embeddings audio.
    Entrée : tenseur de features audio (MFCC, log-mel, etc.)
             taille : (batch_size, time_steps, n_features)
    Sortie : embedding vector de taille (batch_size, embedding_dim)
    """

    def __init__(self, input_dim=40, hidden_dim=128, num_layers=2, embedding_dim=64, dropout=0.3):
        super(AudioNet, self).__init__()

        # Bloc récurrent : LSTM pour capturer la dynamique temporelle
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  # meilleur pour capter contexte avant/après
        )

        # Projection vers un embedding compact
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),  # *2 car bidirectionnel
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x : (batch, time, features)
        """
        # Sorties LSTM
        out, (h, c) = self.lstm(x)

        # On prend le dernier état caché des deux directions
        h_final = torch.cat((h[-2], h[-1]), dim=1)  # concat forward + backward

        # Passage dans le fully connected
        embedding = self.fc(h_final)
        return embedding


# -----------------------------------------------
# Test rapide
# -----------------------------------------------
if __name__ == "__main__":
    model = AudioNet(input_dim=40)
    dummy_audio = torch.randn(8, 100, 40)  # (batch, time, n_mfcc)
    out = model(dummy_audio)
    print("✅ Sortie AudioNet:", out.shape)  # attendu : [8, 64]
