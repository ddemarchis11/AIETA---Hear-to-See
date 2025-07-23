import folder_paths
from ..utils.chunker_utils import chunk_on_silence, fix_split, saliency_split, prosody_split, mfcc_split
from pathlib import Path

class ChunkAudio:
    @classmethod
    def INPUT_TYPES(cls):
        methods = ["chunk_on_silence", "fix_split", "saliency_split", "prosody_split", "mfcc_split"]
        return {"required": {
            "file": ("FILE",),
            "method": (methods, {"default": "chunk_on_silence", "tooltip": "Metodo di chunking"}),  
        }}
    CATEGORY = "audio"
    RETURN_TYPES = ("FOLDER",)
    RETURN_NAMES = ("chunks_folder",)
    FUNCTION = "chunk"

    def chunk(self, file, method):
        path = folder_paths.get_annotated_filepath(file)
        temp_root = Path(folder_paths.get_temp_directory())
        base = Path(path).stem
        out_dir = temp_root / f"{base}_chunks"
        # Chiama il metodo scelto
        if method == "chunk_on_silence":
            chunk_on_silence(path, str(out_dir))
        elif method == "fix_split":
            fix_split(path, str(out_dir))
        elif method == "saliency_split":
            saliency_split(path, str(out_dir))
        elif method == "prosody_split":
            prosody_split(path, str(out_dir))
        elif method == "mfcc_split":
            mfcc_split(path, str(out_dir))
        return (str(out_dir),)

    @classmethod
    def IS_CHANGED(cls, file, method):
        return f"{file}-{method}"
