import shutil
import subprocess
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# Esegue separazione stems con Demucs e restituisce percorso del file vocals
# model_name: nome modello Demucs, device: 'cpu' o 'cuda'
def separate_vocals(audio_path: str, model_name: str, device: str, temp_dir: str) -> str:
    audio = Path(audio_path)
    temp = Path(temp_dir)
    demucs_tmp = temp / "demucs_tmp"
    # Pulisce tmp
    if demucs_tmp.exists(): shutil.rmtree(demucs_tmp)
    demucs_tmp.mkdir(parents=True, exist_ok=True)
    # Comando Demucs
    cmd = [
        "demucs",
        "-n", model_name,
        "--two-stems", "vocals",
        "--device", device,
        "-o", str(demucs_tmp),
        str(audio)
    ]
    subprocess.run(cmd, check=True)
    # Seleziona vocals
    stems = list(demucs_tmp.rglob(f"*_vocals.wav"))
    if not stems:
        raise FileNotFoundError(f"No vocals stem in {demucs_tmp}")
    selected = stems[0]
    # Copia in temp
    out_path = temp / f"{audio.stem}_vocals{audio.suffix}"
    shutil.copy(selected, out_path)
    # Rimuovi tmp
    shutil.rmtree(demucs_tmp)
    return str(out_path)

def spectral_subtraction(audio_path: str, temp_dir: str,
                         n_fft: int = 1024,
                         window: str = 'hann',
                         threshold_factor: float = 0.3,
                         over_sub: float = 5.0,
                         abs_floor: float = 0.01) -> str:
    """ 
    Applica la sottrazione spettrale per rimuovere il rumore di fondo dall'audio.
    
    La sottrazione spettrale è una tecnica di noise reduction che:
    1. Stima lo spettro del rumore dai segmenti "silenziosi"
    2. Sottrae questo rumore stimato da tutto l'audio
    3. Ricostruisce il segnale pulito
    
    Args:
        audio_path (str): Percorso del file audio da pulire
        temp_dir (str): Directory temporanea per il file di output
        
        n_fft (int, default=1024): 
            Dimensione della FFT (Fast Fourier Transform).
            - Valori più alti = maggiore risoluzione in frequenza, ma più lenti
            - Valori più bassi = elaborazione più veloce, ma meno precisa
            - Tipicamente potenze di 2: 512, 1024, 2048, 4096
            - Influenza la qualità della separazione rumore/voce
            
        window (str, default='hann'): 
            Tipo di finestra per la STFT (Short-Time Fourier Transform).
            - 'hann': Finestra di Hanning, buon compromesso (più comune)
            - 'hamming': Finestra di Hamming, riduce le perdite spettrali
            - 'blackman': Finestra di Blackman, migliore soppressione lobi laterali
            - Influenza la qualità della ricostruzione audio
            
        threshold_factor (float, default=0.3):
            Fattore moltiplicativo per la soglia di rilevamento voce.
            - Valore finale: mediana_energia × threshold_factor
            - Valori bassi (0.1-0.4): Più sensibile, rileva più segmenti come voce
            - Valori alti (0.5-1.0): Più selettivo, classifica meno segmenti come voce
            - CRITICO: Se troppo alto, la voce debole viene trattata come rumore
            - Se troppo basso, il rumore viene incluso nella stima della voce
            
        over_sub (float, default=5.0):
            Fattore di sovra-sottrazione del rumore.
            - Formula: spettro_pulito = spettro_originale - (over_sub × rumore_stimato)
            - Valore 1.0 = sottrazione esatta del rumore stimato
            - Valori > 1.0 = sottrazione aggressiva (rimuove più rumore)
            - ATTENZIONE: Valori alti (>3.0) causano artefatti "metallici"
            - Compromesso: più pulizia vs più artefatti
            - Per trascrizione: usare valori bassi (1.2-2.0)
            
        abs_floor (float, default=0.01):
            Soglia minima assoluta per i coefficienti spettrali.
            - Previene coefficienti zero/negativi che causerebbero distorsioni
            - Valore come frazione dell'ampiezza massima (0.01 = 1%)
            - Valori troppo bassi: Possibili "buchi" nello spettro
            - Valori troppo alti: Riduce eccessivamente la rimozione del rumore
            - Preserva le componenti deboli ma importanti della voce
    
    Returns:
        str: Percorso del file audio pulito salvato
        
    """
    audio = Path(audio_path)
    temp = Path(temp_dir)
    hop_length = n_fft // 4
    # Carica audio
    y, sr = librosa.load(str(audio), sr=None)
    # STFT
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=n_fft, window=window, center=False)
    magnitude, phase = np.abs(S), np.angle(S)
    # Calcola energia dei frame
    frames = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)
    frame_energy = np.sum(frames**2, axis=0)
    threshold = np.median(frame_energy) * threshold_factor
    speech_mask = frame_energy > threshold
    # Calcola spettro noise medio
    noise_spectrum = magnitude[:, ~speech_mask]
    noise_mean = np.mean(noise_spectrum, axis=1, keepdims=True)
    # Sottrazione spettrale
    mag_clean = magnitude - over_sub * noise_mean
    mag_clean = np.maximum(mag_clean, abs_floor)
    S_clean = mag_clean * np.exp(1j * phase)
    y_clean = librosa.istft(S_clean, hop_length=hop_length,
                            win_length=n_fft, window=window, center=False)
    # Salva file pulito
    out_path = temp / f"{audio.stem}_cleaned{audio.suffix}"
    sf.write(str(out_path), y_clean, sr)
    return str(out_path)
