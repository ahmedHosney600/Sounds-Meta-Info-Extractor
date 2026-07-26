"""extractors/filename_parser.py — Smart filename parser with no fixed-format assumption."""

import re
from pathlib import Path


def parse_filename(filepath: str) -> dict:
    """
    Extract meaning from the filename without assuming any fixed naming convention.

    Strategy
    --------
    1.  Split by underscore and by dash, choose the dominant separator.
    2.  CamelCase-split each part to extract individual words.
    3.  If ≥3 parts are found, guess Category / Description / Suffix.
    4.  Detect keyword flags (loop, one-shot, BPM hint, key hint).

    All returned keys will be prefixed with ``fn_`` by the caller in main.py.

    Returns
    -------
    dict with keys:
        raw_stem, parts_underscore, parts_dash, detected_separator,
        dominant_parts, part_count, name_words,
        parsed_category, parsed_description, parsed_suffix,
        numbers_in_name, name_bpm_hint, name_key_hint,
        flag_has_loop, flag_has_oneshot, flag_has_bpm, flag_has_key, is_all_caps
    """
    stem: str = Path(filepath).stem

    # ── Split by separators ──────────────────────────────────────────────────
    parts_us   = [p.strip() for p in stem.split("_") if p.strip()]
    parts_dash = [p.strip() for p in stem.split("-") if p.strip()]

    n_us, n_dash = len(parts_us), len(parts_dash)
    if n_us > 1 and n_us >= n_dash:
        sep, parts = "_", parts_us
    elif n_dash > 1:
        sep, parts = "-", parts_dash
    else:
        sep, parts = "none", [stem]

    # ── Individual words (for full-text search) ──────────────────────────────
    words_str = re.sub(r"[_\-]", " ", stem)
    words_str = re.sub(r"([a-z])([A-Z])", r"\1 \2", words_str)  # CamelCase
    name_words = [w.lower() for w in words_str.split() if len(w) > 1]

    # ── Category / Description / Suffix detection ────────────────────────────
    # Heuristic: first part ≤14 chars with no spaces → likely a category tag
    if len(parts) >= 3:
        parsed_cat  = parts[0] if len(parts[0]) <= 14 else None
        parsed_desc = " ".join(parts[1:-1])
        parsed_sfx  = parts[-1]
    elif len(parts) == 2:
        parsed_cat  = parts[0] if len(parts[0]) <= 14 else None
        parsed_desc = parts[1]
        parsed_sfx  = None
    else:
        parsed_cat  = None
        parsed_desc = stem
        parsed_sfx  = None

    # ── Numbers embedded in the name ─────────────────────────────────────────
    numbers = [int(n) for n in re.findall(r"\b\d+\b", stem)]

    # ── Keyword flags ─────────────────────────────────────────────────────────
    lower = stem.lower()
    flag_loop    = any(k in lower for k in ["loop", "_lp", "-lp", "looped"])
    flag_oneshot = any(k in lower for k in ["oneshot", "one_shot", "one-shot", "_hit", "stab"])
    flag_bpm     = bool(re.search(r"\b\d{2,3}\s?bpm\b", lower))
    flag_key     = bool(re.search(r"\b[a-g][#b]?(maj|min|m|major|minor)?\b", lower))

    bpm_m = re.search(r"\b(\d{2,3})\s?bpm\b", lower)
    key_m = re.search(r"\b([a-g][#b]?(?:maj|min|m|major|minor)?)\b", lower)

    return {
        "raw_stem":           stem,
        "parts_underscore":   parts_us,
        "parts_dash":         parts_dash,
        "detected_separator": sep,
        "dominant_parts":     parts,
        "part_count":         len(parts),
        "name_words":         name_words,
        "parsed_category":    parsed_cat,
        "parsed_description": parsed_desc,
        "parsed_suffix":      parsed_sfx,
        "numbers_in_name":    numbers,
        "flag_has_loop":      flag_loop,
        "flag_has_oneshot":   flag_oneshot,
        "flag_has_bpm":       flag_bpm,
        "flag_has_key":       flag_key,
        "name_bpm_hint":      int(bpm_m.group(1)) if bpm_m else None,
        "name_key_hint":      key_m.group(1) if key_m else None,
        "is_all_caps":        stem == stem.upper(),
    }
