"""utils/output_writer.py — Save extraction results to JSON, CSV, Excel, and log."""

import json
import os
import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import config


# Columns shown first in the CSV/Excel — everything else follows alphabetically
_PRIORITY_COLUMNS = [
    "filename", "extension", "parent_folder", "relative_path",
    "file_size_mb", "date_modified",
    "sf_duration_seconds", "sf_duration_formatted",
    "sf_sample_rate", "sf_channels", "sf_bit_depth", "sf_channel_label",
    "sf_format", "sf_subtype",
    "lb_tempo_bpm", "lb_onset_count", "lb_rms_db_mean", "lb_peak_amplitude_db",
    "lb_spectral_centroid_mean", "lb_harmonic_ratio", "lb_percussive_ratio",
    "lb_chroma_dominant_note", "lb_dynamic_range_db",
    "ai_top_class", "ai_top_score", "ai_top5_classes",
    "heuristic_sound_type",
    "heuristic_is_percussive", "heuristic_is_tonal",
    "heuristic_is_loopable", "heuristic_is_one_shot", "heuristic_is_bright",
    "fn_parsed_category", "fn_parsed_description", "fn_parsed_suffix",
    "fn_name_words", "fn_name_bpm_hint", "fn_name_key_hint",
    "tag_title", "tag_artist", "tag_album", "tag_genre",
    "tag_bpm", "tag_key", "tag_comment", "tag_copyright",
    "drive_preview_url", "drive_download_url", "drive_file_id",
    "sha256_hash",
]


def _flatten_record(record: dict) -> dict:
    """Convert lists and dicts to JSON strings so they fit in a CSV cell."""
    row: dict[str, Any] = {}
    for k, v in record.items():
        if isinstance(v, (list, dict)):
            row[k] = json.dumps(v, ensure_ascii=False, default=str)
        elif v is None:
            row[k] = ""
        else:
            row[k] = v
    return row


def save_outputs(
    all_results: list[dict],
    fatal_errors: list[dict],
) -> None:
    """
    Write all_results to:
      - sounds.json   (full nested structure — best for web app)
      - sounds.csv    (flat, lists → JSON strings)
      - sounds.xlsx   (same as CSV)
      - extraction_log.txt
    """
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────
    with open(config.OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    json_mb = os.path.getsize(config.OUTPUT_JSON) / (1024 * 1024)
    print(f"✅  JSON  → {config.OUTPUT_JSON}  ({json_mb:.2f} MB)")

    # ── CSV / Excel ───────────────────────────────────────────────────────────
    flat = [_flatten_record(r) for r in all_results]
    df = pd.DataFrame(flat)

    # Sort columns so priority fields appear first
    other_cols = sorted(c for c in df.columns if c not in _PRIORITY_COLUMNS)
    ordered_cols = [c for c in _PRIORITY_COLUMNS if c in df.columns] + other_cols
    df = df[ordered_cols]

    df.to_csv(config.OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅  CSV   → {config.OUTPUT_CSV}  ({len(df.columns)} columns, {len(df)} rows)")

    df.to_excel(config.OUTPUT_EXCEL, index=False, engine="openpyxl")
    print(f"✅  Excel → {config.OUTPUT_EXCEL}")

    # ── Error log ─────────────────────────────────────────────────────────────
    partial_errors = [r for r in all_results if r.get("_errors")]
    with open(config.OUTPUT_LOG, "w", encoding="utf-8") as f:
        f.write("Sound Metadata Extraction Log\n")
        f.write(f"Run date      : {datetime.datetime.now().isoformat()}\n")
        f.write(f"Source folder : {config.SOUNDS_FOLDER}\n")
        f.write(f"Total files   : {len(all_results)}\n")
        f.write(f"Fatal errors  : {len(fatal_errors)}\n")
        f.write(f"Partial errors: {len(partial_errors)}\n")
        f.write("\n" + "=" * 60 + "\n")
        if fatal_errors:
            f.write("\n--- FATAL ERRORS (file not processed) ---\n\n")
            for err in fatal_errors:
                f.write(f"FILE   : {err['file']}\n")
                f.write(f"ERROR  : {err['error']}\n")
                f.write(f"{err.get('traceback', '')}\n\n")
        if partial_errors:
            f.write("\n--- PARTIAL ERRORS (some fields missing) ---\n\n")
            for r in partial_errors:
                f.write(f"{r.get('filename', '?')}\n")
                for module, msg in r.get("_errors", {}).items():
                    f.write(f"  [{module}] {msg}\n")
                f.write("\n")
    print(f"✅  Log   → {config.OUTPUT_LOG}")
    print(f"\n🎉  All outputs saved to {config.OUTPUT_FOLDER}")
