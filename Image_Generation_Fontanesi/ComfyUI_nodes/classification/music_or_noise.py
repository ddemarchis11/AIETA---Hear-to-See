import os
import folder_paths
from ..utils.music_or_noise_utils import classify_background, defaults

class ClassifyBackgroundAudio:
    LAST_MODEL: str | None = None

    @classmethod
    def INPUT_TYPES(cls):
        models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ("models"))
        files = os.listdir(models_dir)
        joblibs = sorted([f for f in files if f.lower().endswith('.joblib')])
        default_model = cls.LAST_MODEL if cls.LAST_MODEL in joblibs else (joblibs[0] if joblibs else "")
        joblibs = sorted([f for f in files if f.lower().endswith('.joblib')])
        # default sul modello usato l'ultima volta
        default = cls.LAST_MODEL if cls.LAST_MODEL in joblibs else (joblibs[0] if joblibs else "")
        return {"required": {
            "file": ("FILE",),
            "model_file": (joblibs, {"default": default_model, "tooltip": "Seleziona modello background (.joblib)"}),
            "sampling_rate": ("INT", {"default": defaults['sampling_rate'], "min": 8000, "max": 48000, "tooltip": "Frequenza di campionamento audio"}),
            "mono": ("BOOLEAN", {"default": defaults['mono'], "tooltip": "Audio mono o stereo"})
        }}

    CATEGORY = "audio"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("label",)
    FUNCTION = "classify"

    def classify(self, file, model_file, sampling_rate, mono):
        # Persisti scelta del modello
        ClassifyBackgroundAudio.LAST_MODEL = model_file
        path = folder_paths.get_annotated_filepath(file)
        models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ("models"))
        # Carica e applica modello
        label :int = classify_background(path, os.path.join(models_dir, model_file), sr=sampling_rate, mono=mono)
        return (label,)

    @classmethod
    def IS_CHANGED(cls, file, model_file, sampling_rate, mono):
        # Ricalcola se cambia file o modello
        return f"{file}-{model_file}-{sampling_rate}-{mono}"