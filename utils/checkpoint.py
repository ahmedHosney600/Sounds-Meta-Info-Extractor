"""utils/checkpoint.py — Save and load extraction progress to disk."""

import json
import os
from pathlib import Path


def load_checkpoint(checkpoint_file: str) -> dict:
    """Load an existing checkpoint dict (filename → metadata).
    Returns an empty dict if the file does not exist.
    """
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"♻️   Resumed checkpoint — {len(data)} files already processed")
            return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️   Could not load checkpoint ({e}), starting fresh")
    return {}


def save_checkpoint(checkpoint: dict, checkpoint_file: str) -> None:
    """Persist the checkpoint dict to disk (atomic write via temp file)."""
    os.makedirs(Path(checkpoint_file).parent, exist_ok=True)
    tmp = checkpoint_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, default=str)
    os.replace(tmp, checkpoint_file)
