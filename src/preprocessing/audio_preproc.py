import os
import librosa
import numpy as np
from pathlib import Path
import soundfile as sf

# Dossiers
INPUT_DIR = "data/RAVDESS"
OUT_DIR = "data_preprocessed/audio_features"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Paramètres d'extraction
SR_TARGET = 16000
N_MELS = 64
N_MFCC = 40
HOP_LENGTH = 512
WIN_LENGTH = 1024

def process_audio_file(file_path):
    """Charge un fichier audio et extrait ses features."""
    y, sr = sf.read(file_path)

    # Conversion en float et resampling
    if sr != SR_TARGET:
        y = librosa.resample(y.astype(float), orig_sr=sr, target_sr=SR_TARGET)
        sr = SR_TARGET

    # Si stéréo -> mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Spectrogramme mel
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # MFCC
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)

    # Normalisation
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)

    # Concaténation (log_mel + mfcc)
    features = np.concatenate([log_mel, mfcc], axis=0)

    return features.astype(np.float32)


def preprocess_all():
    """Boucle sur tous les fichiers du dossier RAVDESS."""
    for root, _, files in os.walk(INPUT_DIR):
        for f in files:
            if f.lower().endswith(".wav"):
                path = os.path.join(root, f)
                out_path = os.path.join(OUT_DIR, f + ".npy")
                try:
                    feat = process_audio_file(path)
                    np.save(out_path, feat)
                    print("✅ Sauvé :", out_path)
                except Exception as e:
                    print("⚠️ Erreur sur", path, ":", e)


if __name__ == "__main__":
    preprocess_all()
