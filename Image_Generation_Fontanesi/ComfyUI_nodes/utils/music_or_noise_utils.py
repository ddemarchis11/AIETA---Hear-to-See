import os
import numpy as np
import librosa
import joblib
from typing import Tuple
import tensorflow as tf

defaults = {
    'sampling_rate': 16000,
    'mono': True,
}

# Carica modello di background (PCA + classifier)
def load_bg_model(model_file: str) -> Tuple:
    data = joblib.load(model_file)
    return data['pca'], data['clf']



# Classifica background
def classify_background(path: str, model_file: str, sr:int, mono:bool) -> Tuple[int, np.ndarray]:
    # Estrazione embedding YAMNet
    waveform, _ = librosa.load(path, sr=sr, mono=mono)
    yamnet = tf.saved_model.load("E:/ComfyUi/ComfyUI_windows_portable/ComfyUI/custom_nodes/Ai_Arts/classification/models/YAMNET/")
    _, embeddings, _ = yamnet(waveform)
    feat = np.mean(embeddings.numpy(), axis=0).reshape(1, -1)

    pca, clf = load_bg_model(model_file)
    feat_pca = pca.transform(feat)
    label: int = clf.predict(feat_pca)[0]
    return label