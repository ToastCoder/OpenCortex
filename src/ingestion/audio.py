# OpenCortex
# src/ingestion/audio.py — Speech-to-text via whisper-cpp.
# Converts uploaded audio to 16 kHz WAV with ffmpeg, then transcribes with
# the whisper-cpp compiled binary. Only available inside the Docker container.

import os
import shutil
import subprocess
import tempfile

from utils.logger import setup_logger

logger = setup_logger("audio_extractor")

MODEL_PATH = "/models/ggml-tiny.bin"


def check_audio_available():
    """Return whether the whisper-cpp binary and model file are present."""
    if not shutil.which("whisper-cpp"):
        return False, "whisper-cpp binary not in container"
    if not os.path.exists(MODEL_PATH):
        return False, f"model not found at {MODEL_PATH}"
    return True, "whisper-cpp ready"


def process_audio(audio_bytes, source_name):
    """
    Transcribe an audio file to text.

    Pipeline: tempfile → ffmpeg (16 kHz mono WAV) → whisper-cpp → structured block.
    Temporary files are cleaned up in the finally block.
    """
    _, ext = os.path.splitext(source_name)
    raw_path = None
    wav_path = None
    try:
        # Write uploaded bytes to a temp file with the original extension
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(audio_bytes)
            raw_path = f.name

        wav_path = raw_path + ".wav"

        # Normalise to 16 kHz, mono, 16-bit PCM WAV
        ffmpeg_proc = subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1",
             "-sample_fmt", "s16", wav_path],
            capture_output=True,
            text=True,
        )
        if ffmpeg_proc.returncode != 0:
            logger.error(f"ffmpeg failed for {source_name}: {ffmpeg_proc.stderr}")
            return f"\n[Audio Element: {source_name} — ffmpeg conversion failed]\n"

        # Transcribe with whisper-cpp (no timestamps, no Gzip output)
        whisper_proc = subprocess.run(
            ["whisper-cpp", "-f", wav_path, "-m", MODEL_PATH, "-nt", "-ng"],
            capture_output=True,
            text=True,
        )
        if whisper_proc.returncode != 0:
            logger.error(
                f"whisper-cpp failed for {source_name}: {whisper_proc.stderr}"
            )
            return f"\n[Audio Element: {source_name} — transcription failed]\n"

        transcription = whisper_proc.stdout.strip()
        if not transcription:
            transcription = "[no speech detected]"

        logger.info(f"Transcribed {source_name} ({len(transcription)} chars)")

        return (
            f"\n[Audio Element: {source_name}]\n"
            f"Transcription: {transcription}\n"
            f"End of audio transcription for {source_name}.\n"
        )

    except Exception as e:
        logger.error(f"Audio Pipeline Error for {source_name}: {e}")
        return f"\n[Audio Element: {source_name} — error: {e}]\n"
    finally:
        for p in (raw_path, wav_path):
            if p is not None:
                try:
                    os.unlink(p)
                except Exception:
                    pass
