import folder_paths
from ..utils.processing_utils import separate_vocals, spectral_subtraction

import logging


class ProcessAudioByLabel:
    @classmethod
    def INPUT_TYPES(cls):
        demucs_models = ["htdemucs", "mdx_extra", "tasnet"]
        return {"required": {
            "label": ("INT",),
            "file": ("FILE",)}
        , "optional": {
            "demucs_model": (demucs_models, {"default": "htdemucs"}),
            "demucs_device": (["cpu", "cuda"], {"default": "cpu"}),
            "spectral_fft_n": ("INT", {"default": 4096, "min": 64, "max": 8192, "step": 64, "tooltip": "Numero di campioni per FFT"}),
            "spectral_window": (["hann", "hamming", "blackman"], {"default": "hann"}),
            "spectral_threshold_factor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Soglia per rilevamento voce - valori più bassi = più sensibile"}),
        "spectral_over_sub": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 10.0, "step": 0.1, "tooltip": "Fattore sottrazione spettrale - CRITICO: valori alti = più artefatti"}),
        "spectral_abs_floor": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 0.5, "step": 0.001, "tooltip": "Soglia minima spettrale - preserva componenti deboli della voce"})
        }}
    CATEGORY = "audio"
    RETURN_TYPES = ("FILE",)
    RETURN_NAMES = ("cleaned_file",)
    FUNCTION = "process"

    def process(self, label, file, demucs_model, demucs_device, spectral_fft_n,
                         spectral_window, spectral_threshold_factor, spectral_over_sub, spectral_abs_floor ):
        path = folder_paths.get_annotated_filepath(file)
        temp_dir = folder_paths.get_temp_directory()
        if label == 0: # Musica
            # Usa util Demucs
            logging.info(f"Separating vocals from {path} using {demucs_model} on {demucs_device}")
            cleaned = separate_vocals(path, demucs_model, demucs_device, temp_dir)
        else: #rumore 
            #Usa sottrazione spettrale
            logging.info(f"Applying spectral subtraction to {path} with FFT size {spectral_fft_n} and window {spectral_window}")
            cleaned = spectral_subtraction(path, temp_dir,
                                           n_fft=spectral_fft_n,
                                           window=spectral_window,
                                           threshold_factor = spectral_threshold_factor,
                                           over_sub = spectral_over_sub,
                                           abs_floor = spectral_abs_floor)
        return (cleaned,)

    @classmethod
    def IS_CHANGED(cls, label, file, demucs_model, demucs_device, spectral_fft_n, spectral_window, spectral_threshold_factor, spectral_over_sub, spectral_abs_floor):
        return f"{label}-{file}-{demucs_model}-{demucs_device}-{spectral_fft_n}-{spectral_window}-{spectral_threshold_factor}-{spectral_over_sub}-{spectral_abs_floor}"
