# src/train/audio_dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    """
    Dataset pour audio uniquement (RAVDESS)
    """

    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self.files = [f for f in os.listdir(audio_dir) if f.endswith(".npy")]

        # Mapping RAVDESS code → label 0-6
        self.emotion_map = {1:4, 2:4, 3:3, 4:5, 5:0, 6:2, 7:1, 8:6}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        path = os.path.join(self.audio_dir, file)
        x = np.load(path)
        x = torch.tensor(x, dtype=torch.float32)

        # Extraire le code émotion depuis le nom du fichier RAVDESS
        code = int(file.split("-")[2])
        y = torch.tensor(self.emotion_map.get(code, 4), dtype=torch.long)
        return x, y
