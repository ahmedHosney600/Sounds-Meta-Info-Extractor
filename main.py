#!/usr/bin/env python3
"""
main.py — Sound Metadata Extractor
===================================
Entry point for the Sound Meta Extractor project.
"""

import os
import sys
import json
import traceback
from collections import Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── make sure the project root is on sys.path ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
from tqdm import tqdm

import config
from utils.file_discovery import find_audio_files
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.output_writer import save_outputs
from extractors.file_meta import get_file_metadata
from extractors.filename_parser import parse_filename
from extractors.audio_technical import get_audio_technical
from extractors.librosa_features import get_librosa_features
from extractors.mutagen_tags import get_embedded_tags
from extractors.yamnet_classifier import YAMNetClassifier
from extractors.drive_links import get_drive_links
from extractors.heuristics import compute_heuristics


# ── Global YAMNet instance (loaded once per process if using multiprocessing) ──
_yamnet = None

def _get_yamnet():
    global _yamnet
    if _yamnet is None:
        _yamnet = YAMNetClassifier()
    return _yamnet


# ── Per-file extraction ────────────────────────────────────────────────────────

def extract_all_metadata(
    filepath: str,
    drive_id_map: dict | None,
) -> dict:
    """
    Run every extractor on *filepath* and merge results into a single flat dict.
    This runs in a worker process.
    """
    meta: dict = {"_status": "ok", "_errors": {}}
    lb: dict   = {}
    ai: dict   = {}
    fn_parsed: dict = {}

    # 1 ── File-level metadata (OS + SHA256 hash) ──────────────────────────────
    try:
        file_meta = get_file_metadata(filepath)
        meta.update(file_meta)
    except Exception as exc:
        meta["_errors"]["file_meta"] = str(exc)
        meta["filename"] = Path(filepath).name

    # 2 ── Smart filename parsing ───────────────────────────────────────────────
    try:
        fn_parsed = parse_filename(filepath)
        meta.update({f"fn_{k}": v for k, v in fn_parsed.items()})
    except Exception as exc:
        meta["_errors"]["filename_parse"] = str(exc)

    # 3 ── Audio technical specs (soundfile) ───────────────────────────────────
    try:
        meta.update(get_audio_technical(filepath))
    except Exception as exc:
        meta["_errors"]["soundfile"] = str(exc)

    # 4 ── Embedded tags (mutagen) ─────────────────────────────────────────────
    try:
        meta.update(get_embedded_tags(filepath))
    except Exception as exc:
        meta["_errors"]["mutagen"] = str(exc)

    # 5 ── Audio Feature Extraction & AI Classification ────────────────────────
    # Optimize: load audio once at 22050 Hz (for librosa), then resample for YAMNet
    try:
        # Load for librosa
        y, sr = librosa.load(filepath, sr=config.LIBROSA_SR, mono=True, res_type="kaiser_fast")
        
        try:
            lb = get_librosa_features(y=y, sr=sr, n_mfcc=config.N_MFCC)
            meta.update(lb)
        except Exception as exc:
            meta["_errors"]["librosa"] = str(exc)

        # Resample for YAMNet (16000 Hz)
        try:
            yamnet = _get_yamnet()
            if sr != config.YAMNET_SR:
                y_yamnet = librosa.resample(y, orig_sr=sr, target_sr=config.YAMNET_SR, res_type="kaiser_fast")
            else:
                y_yamnet = y
            ai = yamnet.classify(y=y_yamnet)
            meta.update(ai)
        except Exception as exc:
            meta["_errors"]["yamnet"] = str(exc)

    except Exception as exc:
        meta["_errors"]["audio_load"] = str(exc)

    # 6 ── Google Drive links ──────────────────────────────────────────────────
    try:
        meta.update(get_drive_links(meta.get("filename", ""), drive_id_map))
    except Exception as exc:
        meta["_errors"]["drive_links"] = str(exc)

    # 7 ── Heuristics (derived flags) ──────────────────────────────────────────
    try:
        meta.update(compute_heuristics(lb, ai, fn_parsed))
    except Exception as exc:
        meta["_errors"]["heuristics"] = str(exc)

    if meta["_errors"]:
        meta["_status"] = "partial" if len(meta) > 10 else "failed"

    return meta


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("🎵  Sound Metadata Extractor (Optimized)")
    print("=" * 68)
    print(f"   Sounds folder  : {config.SOUNDS_FOLDER}")
    print(f"   Output folder  : {config.OUTPUT_FOLDER}")
    print(f"   Recursive scan : {config.RECURSIVE}")
    print(f"   Max file size  : {config.MAX_FILE_SIZE_MB} MB")
    print(f"   MFCC count     : {config.N_MFCC}")
    print(f"   Workers        : {config.NUM_WORKERS}")
    print()

    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    # ── Step 1: Discover audio files ──────────────────────────────────────────
    print(f"🔍  Scanning: {config.SOUNDS_FOLDER}")
    audio_files = find_audio_files(config.SOUNDS_FOLDER)
    if not audio_files:
        print("❌  No audio files found. Check SOUNDS_FOLDER in config.py.")
        sys.exit(1)

    ext_counts = Counter(Path(f).suffix.lower() for f in audio_files)
    print(f"✅  Found {len(audio_files)} audio files:")
    for ext, cnt in sorted(ext_counts.items()):
        print(f"    {ext:<10} {cnt}")

    # ── Step 2: Load checkpoint ───────────────────────────────────────────────
    checkpoint     = load_checkpoint(config.CHECKPOINT_FILE)
    all_results    = list(checkpoint.values())
    processed_names = set(checkpoint.keys())
    to_process     = [f for f in audio_files if Path(f).name not in processed_names]
    print(f"\n📦  {len(to_process)} new files to process "
          f"({len(all_results)} already done)")

    # ── Step 3: Load Drive ID Map ─────────────────────────────────────────────
    drive_id_map = None
    if config.DRIVE_ID_MAP_FILE and os.path.exists(config.DRIVE_ID_MAP_FILE):
        print(f"\n🔗  Loading Drive ID map from {config.DRIVE_ID_MAP_FILE}...")
        try:
            with open(config.DRIVE_ID_MAP_FILE, "r") as f:
                drive_id_map = json.load(f)
            print(f"✅  Loaded {len(drive_id_map)} Drive IDs.")
        except Exception as exc:
            print(f"⚠️   Could not load Drive ID map: {exc}")
    else:
        print("\n⚠️   No Drive ID map found. Drive links will be empty.")
        print("    Run the 'Generate Drive ID Map' cell in the notebook to enable links.")

    # ── Step 4: Main extraction loop (Multiprocessing) ───────────────────────
    fatal_errors: list[dict] = []
    
    if not to_process:
        print("\n✅  All files already processed!")
    else:
        print(f"\n🚀  Starting extraction with {config.NUM_WORKERS} workers...")
        
        with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
            futures = {
                executor.submit(extract_all_metadata, filepath, drive_id_map): filepath
                for filepath in to_process
            }
            
            with tqdm(total=len(to_process), desc="Extracting metadata", unit="file") as pbar:
                for future in as_completed(futures):
                    filepath = futures[future]
                    fname = Path(filepath).name
                    try:
                        result = future.result()
                        all_results.append(result)
                        processed_names.add(fname)
                        checkpoint[fname] = result
                        
                        if len(all_results) % config.CHECKPOINT_EVERY == 0:
                            save_checkpoint(checkpoint, config.CHECKPOINT_FILE)
                    except Exception as exc:
                        tb = traceback.format_exc()
                        fatal_errors.append({"file": filepath, "error": str(exc), "traceback": tb})
                        tqdm.write(f"❌  FATAL: {fname} → {exc}")
                    
                    pbar.update(1)

        # Final checkpoint
        save_checkpoint(checkpoint, config.CHECKPOINT_FILE)

    partial = sum(1 for r in all_results if r.get("_errors"))
    print(f"\n✅  Extraction complete!")
    print(f"    Total processed  : {len(all_results)}")
    print(f"    Partial errors   : {partial}  (some fields missing)")
    print(f"    Fatal errors     : {len(fatal_errors)}  (file skipped entirely)")

    # ── Step 5: Save outputs ──────────────────────────────────────────────────
    print("\n💾  Saving outputs…")
    save_outputs(all_results, fatal_errors)

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    _print_summary(all_results)


def _print_summary(results: list[dict]) -> None:
    """Print extraction statistics to stdout."""
    if not results:
        return

    print("\n" + "=" * 68)
    print("📊  EXTRACTION SUMMARY")
    print("=" * 68)
    print(f"  Files processed        : {len(results)}")

    # Format breakdown
    exts = Counter(r.get("extension", "?") for r in results)
    print("\n  Formats:")
    for ext, cnt in sorted(exts.items()):
        print(f"    {ext:<12} {cnt}")

    # Duration stats
    durs = [r.get("sf_duration_seconds") for r in results if r.get("sf_duration_seconds")]
    if durs:
        print(f"\n  Duration:")
        print(f"    Min   : {min(durs):.3f}s")
        print(f"    Max   : {max(durs):.3f}s")
        print(f"    Mean  : {sum(durs)/len(durs):.3f}s")
        print(f"    Total : {sum(durs)/60:.1f} minutes")

    # Channels
    ch_counts = Counter(r.get("sf_channel_label", "?") for r in results)
    print(f"\n  Channels:")
    for ch, cnt in ch_counts.items():
        print(f"    {str(ch):<12} {cnt}")

    # AI classes
    ai_classes = [r.get("ai_top_class") for r in results if r.get("ai_top_class")]
    if ai_classes:
        top_cls = Counter(ai_classes).most_common(10)
        print(f"\n  Top AI Classes (YAMNet):")
        for cls, cnt in top_cls:
            print(f"    {str(cls):<35} {cnt}")

    # Heuristic sound types
    sound_types = Counter(r.get("heuristic_sound_type") for r in results if r.get("heuristic_sound_type"))
    print(f"\n  Heuristic Sound Types:")
    for st, cnt in sound_types.most_common():
        print(f"    {str(st):<25} {cnt}")

    # Drive links resolved
    n_linked = sum(1 for r in results if r.get("drive_file_id"))
    print(f"\n  Drive links resolved   : {n_linked} / {len(results)}")
    print("=" * 68)


if __name__ == "__main__":
    main()
