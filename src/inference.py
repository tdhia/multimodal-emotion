import torch
import torch.nn as nn
import librosa
import numpy as np
from src.models.audio_model import AudioNet

# Liste des émotions RAVDESS
EMOTIONS = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}


def predict_emotion(audio_path, model_path="audio_model_ravdess.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle
    model = AudioNet(input_dim=40).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Charger et prétraiter le fichier audio
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0).to(device)  # (1, time, features)

    # Passage dans le modèle
    with torch.no_grad():
        emb = model(mfcc)
        logits = nn.Linear(emb.shape[1], 8).to(device)(emb)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    print(f"🔊 Fichier : {audio_path}")
    print(f"🎭 Émotion prédite : {EMOTIONS[pred]} ({probs[0][pred]:.2%} de confiance)")
    return EMOTIONS[pred]


if __name__ == "__main__":
    # Exemple : tester un fichier audio du dossier RAVDESS
    audio_test = "data/RAVDESS/Actor_01/03-02-01-01-01-01-01.wav"
    predict_emotion(audio_test)
