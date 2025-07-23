import folder_paths
import hashlib
import torchaudio
import os


class LoadAudio:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio", "video"])
        return {"required": {"audio": (sorted(files), {"audio_upload": True})}}

    CATEGORY = "audio"

    RETURN_TYPES = ("AUDIO","FILE" )
    FUNCTION = "load"

    def load(self, audio):
        
        audio_path = folder_paths.get_annotated_filepath(audio)
        if not audio_path.lower().endswith(".wav"):
            wav_path = audio_path.rsplit(".",1)[0] + ".wav"
            waveform, sr = torchaudio.load(audio_path)
            torchaudio.save(wav_path, waveform, sr)
            return ({"waveform": waveform.unsqueeze(0), "sample_rate": sr}, wav_path)
        else:
            waveform, sample_rate = torchaudio.load(audio_path)
            audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            return (audio, audio_path)

    @classmethod
    def IS_CHANGED(s, audio):
        image_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True