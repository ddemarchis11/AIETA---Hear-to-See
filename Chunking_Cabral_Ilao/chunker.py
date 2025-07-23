from pydub import AudioSegment, silence
import soundfile as sf
from scipy.ndimage import median_filter
import os
import librosa
import numpy as np


def chunk_on_silence(
    audio_path: str,
    output_folder: str,
    silence_thresh: int = -40,
    min_silence_len: int = 500
) -> None:
    audio = AudioSegment.from_file(audio_path)
    
    chunks: list[AudioSegment] = silence.split_on_silence(
        audio_segment=audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=200
    )

    os.makedirs(output_folder, exist_ok=True)

    for i, chunk in enumerate(chunks):
        out_file = os.path.join(output_folder, f"chunk_{i+1}.wav")
        chunk.export(out_file, format="wav")

def fix_split(
    audio_path: str,
    output_folder: str,
    interval: int = 5
) -> None:
    y, sr = librosa.load(audio_path, sr=None)

    duration = librosa.get_duration(y=y, sr=sr)
    num_chunks = int(duration // interval) + 1

    for i in range(num_chunks):
        start_sample = int(i * interval * sr)
        end_sample = int(min((i + 1) * interval * sr, len(y)))
        chunk = y[start_sample:end_sample]
        sf.write(os.path.join(output_folder, f"chunk_{i+1}.wav"), chunk, sr)

def saliency_split(
    audio_path: str,
    output_folder: str,
    min_silence_len=0.5,
    silence_thresh_db=-30,
    min_chunk_len=0.3
) -> None:
    y, sr = librosa.load(audio_path, sr=None)
    
    # set frame length to 50ms
    frame_length = int(0.05 * sr)
    # set hop length to 10ms
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
    for start, end in silence_boundaries:
        if start - prev_end >= min_chunk_len:
            segments.append((prev_end, start))
        prev_end = end

    if len(y)/sr - prev_end >= min_chunk_len:
        segments.append((prev_end, len(y)/sr))
    
    os.makedirs(output_folder, exist_ok=True)
    for i, (start, end) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = y[start_sample:end_sample]
        sf.write(f"{output_folder}/chunk_{i+1}.wav", chunk, sr)

def prosody_split(
    audio_path: str,
    output_folder: str,
    pitch_delta_factor: float = 3.0,
    energy_delta_factor: float = 3.0,
    min_chunk_len: float = 3.0,
    max_chunk_len: float = 15.0
) -> None:
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

def mfcc_split(
    audio_path: str,
    output_folder: str,
    n_mfcc: int = 13,
    dist_thresh: float = 50,
    min_chunk_len: float = 1.0
) -> None:
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


input_audio_path = "audio/audio-2.wav"
output_folder_path = "chunks"

methods = [
    chunk_on_silence,
    fix_split,
    saliency_split,
    prosody_split
]


for method in methods:
    new_output = os.path.join(output_folder_path, method.__name__)
    os.makedirs(new_output, exist_ok=True)
    method(input_audio_path, new_output)