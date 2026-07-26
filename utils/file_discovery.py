"""utils/file_discovery.py — Scan a folder and return all audio file paths."""

from pathlib import Path
import config


def find_audio_files(
    folder_path: str,
    recursive: bool | None = None,
    extensions: set | None = None,
    max_size_mb: float | None = None,
) -> list[str]:
    """
    Walk *folder_path* and return a sorted list of audio file paths.

    Parameters
    ----------
    folder_path  : Root directory to scan.
    recursive    : If True, descend into subdirectories (default: config.RECURSIVE).
    extensions   : Set of lowercase extensions to accept (default: config.AUDIO_EXTENSIONS).
    max_size_mb  : Skip files larger than this in MB; 0 means no limit (default: config.MAX_FILE_SIZE_MB).

    Returns
    -------
    Sorted list of absolute file path strings.
    """
    if recursive is None:
        recursive = config.RECURSIVE
    if extensions is None:
        extensions = config.AUDIO_EXTENSIONS
    if max_size_mb is None:
        max_size_mb = config.MAX_FILE_SIZE_MB

    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌  Sounds folder not found: {folder_path}")
        return []

    found: list[str] = []
    pattern = "**/*" if recursive else "*"

    for path in sorted(folder.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        if max_size_mb > 0:
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                print(f"⚠️   Skipping (too large {size_mb:.1f} MB): {path.name}")
                continue
        found.append(str(path))

    return found
