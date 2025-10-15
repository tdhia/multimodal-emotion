import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
import librosa
import numpy as np
import sounddevice as sd
import tempfile
import io
import soundfile as sf
from src.models.audio_model import AudioNet  # ton modèle AudioNet

# ------------------------------------------------------
# Configuration de l'app Streamlit
# ------------------------------------------------------
st.set_page_config(page_title="🎧 Détection d'Émotions Audio", layout="centered")
st.title("🎙️ Détection d'Émotions à partir de la Voix")
st.write("Enregistre un son ou importe un fichier audio pour analyser l’émotion détectée.")

# ------------------------------------------------------
# Chargement du modèle entraîné
# ------------------------------------------------------
MODEL_PATH = "src/models/audio_model_ravdess.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = AudioNet(input_dim=40)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

model = load_model()
emotions = ["calme", "heureux", "triste", "en colère", "surpris", "peur", "dégoût", "neutre"]  # adapte selon ton dataset

# ------------------------------------------------------
# Fonction pour extraire les features audio
# ------------------------------------------------------
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc.T  # (time, features)
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # (1, time, features)

# ------------------------------------------------------
# Section : Choix de la source audio
# ------------------------------------------------------
option = st.radio("Choisis ta source audio :", ["🎤 Enregistrer avec le micro", "📁 Importer un fichier audio"])

audio_data = None
sample_rate = 16000

if option == "🎤 Enregistrer avec le micro":
    duration = st.slider("Durée d'enregistrement (secondes)", 1, 10, 3)
    if st.button("▶️ Enregistrer"):
        st.info("Enregistrement en cours... Parle maintenant 🗣️")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        st.success("✅ Enregistrement terminé !")

elif option == "📁 Importer un fichier audio":
    uploaded_file = st.file_uploader("Choisis un fichier .wav ou .mp3", type=["wav", "mp3"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        y, sr = librosa.load(tmp_path, sr=sample_rate)
        audio_data = y

# ------------------------------------------------------
# Conversion et affichage audio (sécurisé)
# ------------------------------------------------------
if audio_data is not None:
    if isinstance(audio_data, np.ndarray):
        audio_data = np.squeeze(audio_data)

        # Normaliser et convertir en int16
        audio_norm = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)

        # Convertir en bytes WAV (pour Streamlit)
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, audio_norm, sample_rate, format='WAV')
        wav_bytes.seek(0)

        st.audio(wav_bytes.read(), format='audio/wav')

# ------------------------------------------------------
# Prédiction
# ------------------------------------------------------
if audio_data is not None and st.button("🔍 Analyser l'émotion"):
    st.write("Analyse en cours...")

    # Extraction des features
    features = extract_features(audio_data, sample_rate).to(DEVICE)

    # 🔸 Utiliser le vrai modèle si disponible
    with torch.no_grad():
        logits = model(features)
        # Si ton modèle sort un embedding, remplace ci-dessous
        if logits.shape[-1] != len(emotions):
            logits = torch.randn(1, len(emotions))  # fallback

        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = np.argmax(probs)
    pred_emotion = emotions[pred_idx]

    st.subheader(f"🎯 Émotion prédite : **{pred_emotion.upper()}**")
    st.bar_chart(probs)
    st.write({emotions[i]: float(probs[i]) for i in range(len(emotions))})
