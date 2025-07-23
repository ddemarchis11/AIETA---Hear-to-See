import os
import numpy as np
import librosa
import joblib
from collections import Counter
from typing import List
import folder_paths

# Default parameters
defaults = {
    'window_seconds': 3.0,
    'sr': 44100,
    'hop_length': 256,
    'n_fft': 1024,
    'n_mfcc': 13
}

# Statistiche su MFCC
def aggregate_stats(feats: np.ndarray) -> np.ndarray:
    out = []
    for row in feats:
        vals = row.astype(np.float32)
        out.extend([
            np.mean(vals),
            np.std(vals),
            np.median(vals),
            np.max(vals) - np.min(vals)
        ])
    return np.asarray(out, dtype=np.float32)

# Estrazione MFCC + delta
def extract_mfcc_from_signal(y: np.ndarray, sr: int, n_mfcc: int, n_fft: int, hop_length: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, center=True)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack([mfcc, delta1, delta2])
    return aggregate_stats(feats)

# Segmentazione audio
def segment_audio(path: str, window_sec: float, sr: int) -> List[np.ndarray]:
    y, _ = librosa.load(path, sr=sr)
    win_len = int(window_sec * sr)
    n_segs = int(np.ceil(len(y) / win_len))
    segments = []
    for i in range(n_segs):
        start = i * win_len
        end = start + win_len
        seg = y[start:end]
        if len(seg) < win_len:
            seg = np.pad(seg, (0, win_len - len(seg)), mode='constant')
        segments.append(seg)
    return segments

# Estrazione feature per segmenti
def extract_features(path: str, window_sec: float, sr: int, n_mfcc: int, n_fft: int, hop_length: int) -> np.ndarray:
    segments = segment_audio(path, window_sec, sr)
    feats = [extract_mfcc_from_signal(seg, sr, n_mfcc, n_fft, hop_length) for seg in segments]
    return np.vstack(feats)

# Classificazione segmenti
def classify_segments(path: str, model, scaler, feat_kwargs: dict) -> List[int]:
    X = extract_features(path, **feat_kwargs)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled).tolist()

# Majority vote
def majority_vote(preds: List[int]) -> int:
    cnt = Counter(preds)
    return cnt.most_common(1)[0][0]

# Caricamento modello
def load_model(model_file: str):
    data = joblib.load(model_file)
    return data['model'], data['scaler']
