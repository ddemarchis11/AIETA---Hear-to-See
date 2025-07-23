# custom_nodes/audio/transcribe_chunks.py
import os
from pathlib import Path
import folder_paths
from ..utils.whisper_utils import transcribe_file_whisper

class TranscribeChunks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chunks_folder": ("FOLDER",),
            "model": (["tiny", "base", "small", "medium", "large-v2"],{"default":"small"}, {"tooltip": "Cartella del modello Whisper scaricato"}),
            "language": (["en", "it", "fr", "de"], {"default": "it"}),
            "task": (["transcribe", "translate"], {"default": "transcribe"}),
            "device": (["cpu", "cuda"], {"default": "cpu", "tooltip": "Dispositivo per eseguire il modello"})
        }}
    CATEGORY = "audio"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("chunk_folder",)
    FUNCTION = "transcribe"

    def transcribe(self, chunks_folder, model, language, task, device):
        temp = folder_paths.get_temp_directory()
        # create output subfolder for this batch
        out_dir = os.path.join(temp, Path(chunks_folder).stem + "_transcripts")
        os.makedirs(out_dir, exist_ok=True)
        # Lista file wav nella cartella
        folder = Path(chunks_folder)
        wavs = sorted(folder.glob("*.wav"))
        for wav in wavs:
            txt = transcribe_file_whisper(
                str(wav), model, language, task, device, out_dir
            )
        return (out_dir,)

    @classmethod
    def IS_CHANGED(cls, chunks_folder, model, language, task, device):
        return f"{chunks_folder}-{model}-{language}-{task}-{device}"
