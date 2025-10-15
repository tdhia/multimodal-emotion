import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import librosa
import numpy as np
from tqdm import tqdm
from src.models.audio_model import AudioNet  # ton modèle AudioNet

# =========================================================
# 🧠 Dataset Audio RAVDESS
# =========================================================
class AudioDataset(Dataset):
    def __init__(self, data_dir="data/RAVDESS", sr=16000, n_mfcc=40):
        self.data_dir = data_dir
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.files = []
        self.labels = []

        # RAVDESS : structure Actor_xx / 03-01-01-01-01-01-01.wav
        for actor_folder in os.listdir(data_dir):
            actor_path = os.path.join(data_dir, actor_folder)
            if not os.path.isdir(actor_path):
                continue
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    path = os.path.join(actor_path, file)
                    self.files.append(path)
                    # label = 3e nombre du nom de fichier (émotion)
                    emotion_id = int(file.split("-")[2])
                    self.labels.append(emotion_id - 1)  # range 0–7

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        y, sr = librosa.load(path, sr=self.sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = mfcc.T  # (time, n_mfcc)

        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# =========================================================
# 🧩 Fonction de padding dynamique
# =========================================================
def collate_fn(batch):
    xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    xs_padded = rnn_utils.pad_sequence(xs, batch_first=True)  # (batch, max_time, n_mfcc)
    ys = torch.stack(ys)
    return xs_padded, ys


# =========================================================
# 🚀 Entraînement
# =========================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        logits = nn.Linear(out.shape[1], 8).to(device)(out)  # 8 émotions RAVDESS
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


# =========================================================
# 🎯 Validation
# =========================================================
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Eval", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = nn.Linear(out.shape[1], 8).to(device)(out)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


# =========================================================
# 🧠 Main
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    dataset = AudioDataset(data_dir="data/RAVDESS")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = AudioNet(input_dim=40).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        print(f"\n🚀 Epoch {epoch+1}/10")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.3f}")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.3f}")

    torch.save(model.state_dict(), "audio_model_ravdess.pth")
    print("✅ Modèle sauvegardé dans audio_model_ravdess.pth")


if __name__ == "__main__":
    main()
