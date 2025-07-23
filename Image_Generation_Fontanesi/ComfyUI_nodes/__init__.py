from .audio.load_audio import LoadAudio
from .classification.vocals_or_not import ClassifyAudio
from .classification.music_or_noise import ClassifyBackgroundAudio
from .conditional.conditional import ConditionalNode
from .audio.clear_audio import ProcessAudioByLabel
from .audio.chunk_audio import ChunkAudio
from .audio.whisper import TranscribeChunks

# Registrazione dei nodi per ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadAudioFile": LoadAudio,
    "ClassifyAudio": ClassifyAudio,
    "ClassifyBackgroundAudio": ClassifyBackgroundAudio,
    "ConditionalNode": ConditionalNode,
    "ClearAudio": ProcessAudioByLabel,
    "ChunkAudio": ChunkAudio,
    "TranscribeChunks": TranscribeChunks

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudioFile": "Load Audio File",
    "ClassifyAudio": "Classify Vocals or Not",
    "ClassifyBackgroundAudio": "Classify Background Audio",
    "ConditionalNode": "Conditional Node",
    "ClearAudio": "Audio Preprocessing Clear",
    "ChunkAudio": "Audio Chunker",
    "TranscribeChunks": "Audio Transcription by Whisper"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']