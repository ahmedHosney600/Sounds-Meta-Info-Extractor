"""
config.py — All user-configurable settings for Sound Meta Extractor.

Edit this file before running, OR override any value via environment variables.
On Google Colab, the Colab launcher notebook sets these for you.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
# Full path to your sounds folder on Google Drive (as mounted in Colab)
# Example: "/content/drive/MyDrive/Sound Libraries/BLOW"
SOUNDS_FOLDER: str = os.environ.get(
    "SOUNDS_FOLDER",
    "/content/drive/MyDrive/SOUNDS"
)

# Google Drive Folder ID — found in the browser URL when you open the folder:
# https://drive.google.com/drive/folders/<FOLDER_ID>
# Set to None or leave empty to skip Drive link generation.
DRIVE_FOLDER_ID: str | None = os.environ.get("DRIVE_FOLDER_ID") or None

# Output directory (created automatically if it doesn't exist)
OUTPUT_FOLDER: str = os.environ.get(
    "OUTPUT_FOLDER",
    "/content/drive/MyDrive/sounds_metadata_output"
)

# ── Scan Settings ──────────────────────────────────────────────────────────────
# Search inside subfolders recursively?
RECURSIVE: bool = os.environ.get("RECURSIVE", "true").lower() == "true"

# Supported audio file extensions (all lowercase)
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

# ── Output File Paths (derived from OUTPUT_FOLDER) ────────────────────────────
CHECKPOINT_FILE: str = os.path.join(OUTPUT_FOLDER, "checkpoint.json")
OUTPUT_JSON: str     = os.path.join(OUTPUT_FOLDER, "sounds.json")
OUTPUT_CSV: str      = os.path.join(OUTPUT_FOLDER, "sounds.csv")
OUTPUT_EXCEL: str    = os.path.join(OUTPUT_FOLDER, "sounds.xlsx")
OUTPUT_LOG: str      = os.path.join(OUTPUT_FOLDER, "extraction_log.txt")
