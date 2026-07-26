"""extractors/file_meta.py — OS-level file metadata + SHA256 hash."""

import datetime
import hashlib
from pathlib import Path

import config


def get_file_metadata(filepath: str) -> dict:
    """
    Extract OS-level file metadata and compute SHA256 hash.

    Fields returned
    ---------------
    filename, stem, extension, parent_folder, relative_path, full_path,
    depth_in_root, file_size_bytes, file_size_kb, file_size_mb,
    date_modified (ISO-8601 UTC), date_created (ISO-8601 UTC), sha256_hash
    """
    path = Path(filepath)
    stat = path.stat()
    size_bytes = stat.st_size

    mtime = datetime.datetime.fromtimestamp(stat.st_mtime, tz=datetime.timezone.utc)
    ctime = datetime.datetime.fromtimestamp(stat.st_ctime, tz=datetime.timezone.utc)

    # SHA256 — stream in 64 KB chunks to handle large files without OOM
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha256.update(chunk)

    # Relative path from the sounds root
    sounds_root = Path(config.SOUNDS_FOLDER)
    try:
        relative = str(path.relative_to(sounds_root))
    except ValueError:
        relative = str(path)

    return {
        "filename":        path.name,
        "stem":            path.stem,
        "extension":       path.suffix.lower(),
        "parent_folder":   path.parent.name,
        "relative_path":   relative,
        "full_path":       str(path),
        "depth_in_root":   len(path.parts) - len(sounds_root.parts),
        "file_size_bytes": size_bytes,
        "file_size_kb":    round(size_bytes / 1024, 2),
        "file_size_mb":    round(size_bytes / (1024 * 1024), 4),
        "date_modified":   mtime.isoformat(),
        "date_created":    ctime.isoformat(),
        "sha256_hash":     sha256.hexdigest(),
    }
