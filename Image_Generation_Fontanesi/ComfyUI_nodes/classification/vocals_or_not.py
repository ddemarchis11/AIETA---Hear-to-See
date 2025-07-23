from typing import List
import folder_paths
import os
from ..utils.vocals_or_not_utils import (
    load_model,
    classify_segments,
    majority_vote,
    defaults
)

class ClassifyAudio:
    LAST_MODEL: str | None = None

    @classmethod
    def INPUT_TYPES(cls):
        models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ("models"))
        files = os.listdir(models_dir)
        joblibs = sorted([f for f in files if f.lower().endswith('.joblib')])
        default_model = cls.LAST_MODEL if cls.LAST_MODEL in joblibs else (joblibs[0] if joblibs else "")
        return {"required": {
            "file": ("FILE",),
            "model_file": (joblibs, {"default": default_model}),
            "window_seconds": ("FLOAT", {"default": defaults['window_seconds'], "min": 0.1, "max": 10.0}),
            "hop_length": ("INT", {"default": defaults['hop_length'], "min": 1, "max": 1024}),
            "n_fft": ("INT", {"default": defaults['n_fft'], "min": 256, "max": 4096}),
            "n_mfcc": ("INT", {"default": defaults['n_mfcc'], "min": 1, "max": 40}),
        }}

    CATEGORY = "audio"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("label",)
    FUNCTION = "classify"

    def classify(self, file, model_file, window_seconds, hop_length, n_fft, n_mfcc):
        ClassifyAudio.LAST_MODEL = model_file
        path = folder_paths.get_annotated_filepath(file)
        model_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", model_file)
        model, scaler = load_model(model_file)
        feat_kwargs = {
            'window_sec': window_seconds,
            'sr': defaults['sr'],
            'n_mfcc': n_mfcc,
            'n_fft': n_fft,
            'hop_length': hop_length,
        }
        preds: List[int] = classify_segments(path, model, scaler, feat_kwargs)
        label: int = majority_vote(preds)
        return (label,)

    @classmethod
    def IS_CHANGED(cls, file, model_file, window_seconds, hop_length, n_fft, n_mfcc):
        return f"{file}-{model_file}-{window_seconds}-{hop_length}-{n_fft}-{n_mfcc}"
