import argparse
import shutil
import subprocess
import os
from pathlib import Path

def separate_stems(
    audio_path: Path,
    model_name: str,
    device: str,
    out_dir: Path,
) -> None:
    # eseguiamo Demucs con 'two stemps' in output e 'vocals' come obiettivo di separazione
    tmp_dir = Path("demucs_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    cmd = [
        "demucs",
        "-n", model_name,
        "--two-stems", "vocals",
        "--device", device,
        "-o", str(tmp_dir),
        str(audio_path)
    ]
    subprocess.run(cmd, check=True)

    stems = list(tmp_dir.rglob("*.wav"))
    if not stems:
        raise FileNotFoundError(f"Nessun file .wav trovato in {tmp_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    base = audio_path.stem 
    for stem in stems:
        # stem.name è "vocals.wav" oppure "no_vocals.wav"
        suffix = stem.suffix   # ".wav"
        tag = stem.stem        # "vocals" o "no_vocals"
        new_name = f"{base}_{tag}{suffix}"
        shutil.copy(stem, out_dir / new_name)

    shutil.rmtree(tmp_dir)

def main():
    out   = Path("stems")
    model = "htdemucs"
    device = "cpu"
    
    traces= ["audio_traces/try_horn.wav"]

    for audio in traces:
        print(f"Separazione stems da {audio} in {out}…")
        separate_stems(Path(audio), model, device, out)
        print("Operazione completata: stems salvate.")

if __name__ == "__main__":
    main()