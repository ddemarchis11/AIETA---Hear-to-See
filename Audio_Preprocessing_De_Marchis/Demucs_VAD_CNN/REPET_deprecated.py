import librosa, numpy as np, scipy.signal as sig, soundfile as sf

def main():
    fname = "noisy_bground.wav"

    y, sr = librosa.load(f"audio_traces/{fname}", sr=44100, mono=True)

    n_fft, hop = 4096, 1024
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    M, P = np.abs(S), np.angle(S)

    M_harm = sig.medfilt(M, kernel_size=(1, 31))

    eps = 1e-10
    mask_bg  = M_harm / (M + eps)
    mask_voc = 1.0 - mask_bg

    S_voc = mask_voc * S
    y_voc = librosa.istft(S_voc, hop_length=hop)

    print("Max abs y_voc:", np.max(np.abs(y_voc)))
    if np.max(np.abs(y_voc)) < 1e-3:
        y_voc /= np.max(np.abs(y_voc))  # normalizzo

    sf.write('noisy_bground.wav', y_voc, sr, subtype='PCM_16')
    print(f"Done: {fname}_repet.wav")
    
if __name__ == "__main__": main()