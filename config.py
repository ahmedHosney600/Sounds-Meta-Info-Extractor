"""
config.py — All user-configurable settings for Sound Meta Extractor.

Edit this file before running, OR override any value via environment variables.
On Google Colab, the Colab launcher notebook sets these for you.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
SOUNDS_FOLDER: str = os.environ.get(
    "SOUNDS_FOLDER",
    "/content/drive/MyDrive/SOUNDS"
)

DRIVE_FOLDER_ID: str | None = os.environ.get("DRIVE_FOLDER_ID") or None

OUTPUT_FOLDER: str = os.environ.get(
    "OUTPUT_FOLDER",
    "/content/drive/MyDrive/sounds_metadata_output"
)

# ── Scan Settings ──────────────────────────────────────────────────────────────
RECURSIVE: bool = os.environ.get("RECURSIVE", "true").lower() == "true"

AUDIO_EXTENSIONS: set = {
    ".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif",
    ".m4a", ".opus", ".wma", ".caf", ".aac",
    ".ac3", ".amr", ".au",
}

# Skip files larger than this (MB). Set 0 for no limit.
MAX_FILE_SIZE_MB: float = float(os.environ.get("MAX_FILE_SIZE_MB", "500"))

# ── Feature Extraction Settings ───────────────────────────────────────────────
# Sample rate for librosa analysis (resamples audio if needed)
LIBROSA_SR: int = int(os.environ.get("LIBROSA_SR", "22050"))

# YAMNet requires exactly 16 kHz mono
YAMNET_SR: int = 16000

# Number of MFCC coefficients to extract
N_MFCC: int = int(os.environ.get("N_MFCC", "20"))

# Number of top YAMNet predictions to keep per file
YAMNET_TOP_K: int = 5

# ── Performance ───────────────────────────────────────────────────────────────
# Number of parallel worker threads for file processing.
# Each worker loads audio + runs librosa on CPU concurrently.
# YAMNet (GPU) is called from the main thread and is thread-safe in TF2.
# Recommended: 2–4 on Colab. Higher values help when Drive I/O is the bottleneck.
NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", "3"))

# Save checkpoint every N files (increase for speed, decrease for safety)
CHECKPOINT_EVERY: int = int(os.environ.get("CHECKPOINT_EVERY", "25"))

# Path to pre-built Drive ID map JSON (built interactively in Colab Cell 3b)
# Set to empty/None to skip Drive links.
DRIVE_ID_MAP_FILE: str = os.environ.get(
    "DRIVE_ID_MAP_FILE",
    "/content/drive_id_map.json"
)

# ── Output File Paths (derived from OUTPUT_FOLDER) ────────────────────────────
CHECKPOINT_FILE: str = os.path.join(OUTPUT_FOLDER, "checkpoint.json")
OUTPUT_JSON: str     = os.path.join(OUTPUT_FOLDER, "sounds.json")
OUTPUT_CSV: str      = os.path.join(OUTPUT_FOLDER, "sounds.csv")
OUTPUT_EXCEL: str    = os.path.join(OUTPUT_FOLDER, "sounds.xlsx")
OUTPUT_LOG: str      = os.path.join(OUTPUT_FOLDER, "extraction_log.txt")
