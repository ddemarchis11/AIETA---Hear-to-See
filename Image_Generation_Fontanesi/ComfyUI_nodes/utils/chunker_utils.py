import os
from pathlib import Path
from typing import Tuple
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment, silence
from scipy.ndimage import median_filter



def chunk_on_silence(audio_path: str, output_folder: str, silence_thresh: int = -40, min_silence_len: int = 600) -> None:
    """
    Divide un file audio in segmenti dove viene rilevato silenzio.
    Parametri:
        audio_path: percorso del file audio di input.
        output_folder: cartella dove salvare i segmenti.
        silence_thresh: soglia di silenzio in dB (default -40).
        min_silence_len: durata minima del silenzio in ms (default 500).
    """
    audio = AudioSegment.from_file(audio_path)
    chunks = silence.split_on_silence(
        audio_segment=audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=200
    )
    os.makedirs(output_folder, exist_ok=True)
    for i, chunk in enumerate(chunks):
        out_file = os.path.join(output_folder, f"chunk_{i+1}.wav")
        chunk.export(out_file, format="wav")


def fix_split(audio_path: str, output_folder: str, interval: float = 5.0) -> None:
    """
    Divide un file audio in segmenti di lunghezza fissa.
    Parametri:
        audio_path: percorso del file audio di input.
        output_folder: cartella dove salvare i segmenti.
        interval: durata di ogni segmento in secondi (default 5).
    """
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    num_chunks = int(duration // interval) + 1
    for i in range(num_chunks):
        start = int(i * interval * sr)
        end = int(min((i + 1) * interval * sr, len(y)))
        chunk = y[start:end]
        os.makedirs(output_folder, exist_ok=True)
        sf.write(os.path.join(output_folder, f"chunk_{i+1}.wav"), chunk, sr)


def saliency_split(audio_path: str, output_folder: str, min_silence_len: float = 0.5, silence_thresh_db: float = -30.0, min_chunk_len: float = 10) -> None:
    """
    Divide un file audio in segmenti basati sui silenzi rilevati tramite l'energia del segnale.
    Parametri:
       audio_path: percorso del file audio di input.
       output_folder: cartella dove salvare i segmenti.
       min_silence_len: durata minima (in secondi) di un silenzio per essere considerato come separatore (default 0.5).
       silence_thresh_db: soglia di energia (in dB) sotto la quale un frame è considerato silenzioso (default -30.0).
       min_chunk_len: durata minima (in secondi) di un segmento esportato (default 10).
    La funzione salva ogni segmento come file WAV nella cartella di output.
    """
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = int(0.05 * sr)
    hop_length = int(0.01 * sr)
    energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)
    silence_frames = np.where(energy_db < silence_thresh_db)[0]
    silence_times = librosa.frames_to_time(silence_frames, sr=sr, hop_length=hop_length)
    silence_boundaries = []
    start = None
    for t in silence_times:
        if start is None:
            start = t
            prev = t
        elif t - prev > 0.05:
            if prev - start >= min_silence_len:
                silence_boundaries.append((start, prev))
            start = t
        prev = t
    if start is not None and prev - start >= min_silence_len:
        silence_boundaries.append((start, prev))
    segments = []
    prev_end = 0
    for st, ed in silence_boundaries:
        if st - prev_end >= min_chunk_len:
            segments.append((prev_end, st))
        prev_end = ed
    if len(y)/sr - prev_end >= min_chunk_len:
        segments.append((prev_end, len(y)/sr))
    os.makedirs(output_folder, exist_ok=True)
    for i, (st, ed) in enumerate(segments):
        ss = int(st * sr)
        ee = int(ed * sr)
        sf.write(os.path.join(output_folder, f"chunk_{i+1}.wav"), y[ss:ee], sr)


def prosody_split(audio_path: str, output_folder: str, pitch_delta_factor: float = 3.0, energy_delta_factor: float = 3.0, min_chunk_len: float = 10.0, max_chunk_len: float = 30.0) -> None:
    """
    Divide l'audio in base a cambiamenti di pitch ed energia (prosodia).
    Parametri:
        audio_path: percorso del file audio di input.
        output_folder: cartella dove salvare i segmenti.
        pitch_delta_factor: sensibilità ai cambiamenti di pitch (default 3.0).
        energy_delta_factor: sensibilità ai cambiamenti di energia (default 3.0).
        min_chunk_len: durata minima di un segmento in secondi (default 3.0).
        max_chunk_len: durata massima di un segmento in secondi (default 15.0).
    """
    y, sr = librosa.load(audio_path, sr=None)
    hop_length = int(0.01 * sr)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
    pitch_track = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        pitch_track.append(pitch)
    pitch_track = np.array(pitch_track)
    mean_pitch = np.mean(pitch_track[pitch_track > 0]) if np.any(pitch_track > 0) else 0
    pitch_track = np.where(pitch_track == 0, mean_pitch, pitch_track)
    pitch_track = median_filter(pitch_track, size=5)

    energy = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)

    pitch_diff = np.abs(np.diff(pitch_track))
    energy_diff = np.abs(np.diff(energy_db))
    minlen = min(len(pitch_diff), len(energy_diff))
    pitch_diff = pitch_diff[:minlen]
    energy_diff = energy_diff[:minlen]

    pitch_thresh = pitch_delta_factor * np.std(pitch_diff)
    energy_thresh = energy_delta_factor * np.std(energy_diff)

    change_points = np.where((pitch_diff > pitch_thresh) | (energy_diff > energy_thresh))[0]
    change_times = librosa.frames_to_time(change_points, sr=sr, hop_length=hop_length)

    filtered_change_times = []
    last_t = 0.0
    for t in change_times:
        if t - last_t >= min_chunk_len:
            filtered_change_times.append(t)
            last_t = t

    segments = []
    prev = 0
    for t in filtered_change_times:
        while t - prev > max_chunk_len:
            segments.append((prev, prev + max_chunk_len))
            prev += max_chunk_len
        if t - prev >= min_chunk_len:
            segments.append((prev, t))
            prev = t
    if len(y)/sr - prev >= min_chunk_len:
        segments.append((prev, len(y)/sr))

    os.makedirs(output_folder, exist_ok=True)
    for i, (start, end) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = y[start_sample:end_sample]
        sf.write(f"{output_folder}/chunk_{i+1}.wav", chunk, sr)


def mfcc_split(audio_path: str, output_folder: str, n_mfcc: int = 13, dist_thresh: float = 50.0, min_chunk_len: float = 10.0) -> None:
    """
    Divide l'audio in base a cambiamenti nei coefficienti MFCC.
    Parametri:
        audio_path: percorso del file audio di input.
        output_folder: cartella dove salvare i segmenti.
        n_mfcc: numero di MFCC da calcolare (default 13).
        dist_thresh: soglia di distanza per la segmentazione (default 50).
        min_chunk_len: durata minima di un segmento in secondi (default 1.0).
    """
    y, sr = librosa.load(audio_path, sr=None)
    hop_length = int(0.01 * sr)  # 10ms hop
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

    # Calcola la distanza euclidea tra frame MFCC adiacenti
    mfcc_diff = np.linalg.norm(np.diff(mfcc, axis=1), axis=0)
    # Soglia: cambia qui per essere più/meno sensibile
    change_points = np.where(mfcc_diff > dist_thresh)[0]
    change_times = librosa.frames_to_time(change_points, sr=sr, hop_length=hop_length)

    # Segmentazione evitando segmenti troppo corti
    segments = []
    prev = 0
    for t in change_times:
        if t - prev >= min_chunk_len:
            segments.append((prev, t))
            prev = t
    if len(y)/sr - prev >= min_chunk_len:
        segments.append((prev, len(y)/sr))

    os.makedirs(output_folder, exist_ok=True)
    for i, (start, end) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = y[start_sample:end_sample]
        sf.write(os.path.join(output_folder, f"chunk_{i+1}.wav"), chunk, sr)