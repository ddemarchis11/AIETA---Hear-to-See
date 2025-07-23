import os
import subprocess
from pathlib import Path
from typing import List
import folder_paths

# Utilità per trascrizione con Whisper via comando

def transcribe_file_whisper(
    audio_path: str,
    model: str,
    language: str,
    task: str,
    device: str,
    temp_dir: str,
) -> str:
    """
    Esegue Whisper CLI su un file audio.
    - model: modello whisper
    - language: lingua del riconoscimento ('en', 'it', etc.)
    - task: 'transcribe' o 'translate'
    - temp_dir: cartella temporanea per output
    Restituisce il path del file di output (testo .txt) in temp_dir.
    """
    audio = Path(audio_path)
    temp = Path(temp_dir)
    temp.mkdir(parents=True, exist_ok=True)
    output_txt = temp / f"{audio.stem}_whisper.txt"
    cmd = [
        "whisper",
        str(audio),
        "--model", model,
        "--language", language,
        "--task", task,
        "--device", device,
        "--output_format", "txt",
        "--output_dir", str(temp)
    ]
    subprocess.run(cmd, check=True)
    # Whisper salva con nome stem.txt
    return str(output_txt)
