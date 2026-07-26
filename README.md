# 🎵 Sound Metadata Extractor

A complete Python project that scans a Google Drive folder of audio files and extracts **maximum metadata** from every file. Designed to run on **Google Colab** connected to Google Drive.

The extracted data is the foundation for a future web-based **advanced sound search and categorisation** interface.

---

## Project Structure

```
Sound Meta Extractor/
├── config.py                  # All settings (paths, sample rates, etc.)
├── main.py                    # Entry point — run this on Colab
├── requirements.txt           # Python dependencies
├── run_on_colab.ipynb         # Thin Colab launcher notebook (5 cells)
│
├── extractors/                # One module per extraction type
│   ├── __init__.py
│   ├── file_meta.py           # OS metadata + SHA256 hash
│   ├── filename_parser.py     # Smart filename parsing (no fixed format assumed)
│   ├── audio_technical.py     # Sample rate, bit depth, channels (soundfile)
│   ├── librosa_features.py    # Full spectral/acoustic/rhythm feature set
│   ├── mutagen_tags.py        # Embedded ID3/Vorbis/RIFF/MP4 tags
│   ├── yamnet_classifier.py   # YAMNet AI classification (521 classes)
│   ├── drive_links.py         # Google Drive preview/download URL generation
│   └── heuristics.py          # Derived flags (percussive, loopable, etc.)
│
└── utils/
    ├── __init__.py
    ├── file_discovery.py      # Recursive audio file scanner
    ├── checkpoint.py          # Save/resume progress to disk
    └── output_writer.py       # Save JSON, CSV, Excel, log
```

---

## How to Run on Google Colab

### Step 1 — Upload the project to Google Drive
Upload this entire `Sound Meta Extractor/` folder to your Google Drive root (`My Drive`).

### Step 2 — Open the Colab launcher
Open `run_on_colab.ipynb` in Google Colab (File → Open → Google Drive).

### Step 3 — Configure (Cell 3 only)
Edit these two lines in **Cell 3**:

```python
os.environ['SOUNDS_FOLDER']   = '/content/drive/MyDrive/SOUNDS'       # your sounds path
os.environ['DRIVE_FOLDER_ID'] = '1AbCdEfGhIjKlMnOpQr'                 # from Drive URL
```

> **Finding your Folder ID:** Open the sounds folder in Google Drive in your browser.  
> The URL will look like: `https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQr`  
> Copy the last part — that's your folder ID.

### Step 4 — Run All Cells

That's it. The outputs will appear in `sounds_metadata_output/` on your Drive.

---

## Outputs

| File | Description |
|---|---|
| `sounds.json` | Full nested metadata — **use this for the web app** |
| `sounds.csv` | Flat table, all list fields JSON-serialised |
| `sounds.xlsx` | Same as CSV in Excel format |
| `extraction_log.txt` | Per-file error report |
| `checkpoint.json` | Progress checkpoint (auto-resumes if interrupted) |

---

## Metadata Fields Extracted (~90+ columns per file)

### 📁 File Info
`filename`, `extension`, `parent_folder`, `relative_path`, `file_size_mb`, `date_modified`, `date_created`, `sha256_hash`

### 🏷️ Filename Parsing (`fn_` prefix)
`fn_parsed_category`, `fn_parsed_description`, `fn_parsed_suffix`, `fn_name_words`, `fn_name_bpm_hint`, `fn_name_key_hint`, `fn_flag_has_loop`, `fn_flag_has_oneshot`

### 🔊 Audio Technical (`sf_` prefix)
`sf_sample_rate`, `sf_channels`, `sf_bit_depth`, `sf_duration_seconds`, `sf_format`, `sf_subtype`, `sf_channel_label`

### 📊 Acoustic Features (`lb_` prefix)
`lb_rms_mean/max/std/db`, `lb_peak_amplitude_db`, `lb_dynamic_range_db`, `lb_zcr_mean`, `lb_spectral_centroid_mean`, `lb_spectral_bandwidth_mean`, `lb_spectral_rolloff_mean`, `lb_spectral_flatness_mean`, `lb_spectral_contrast_mean` (7 bands), `lb_mel_*`, `lb_harmonic_ratio`, `lb_percussive_ratio`, `lb_tempo_bpm`, `lb_beat_count`, `lb_onset_count`, `lb_onset_rate_per_sec`, `lb_chroma_mean` (12 notes), `lb_chroma_dominant_note`, `lb_tonnetz_mean`, `lb_mfcc_01_mean` … `lb_mfcc_20_mean` (+ std for each)

### 🏷️ Embedded Tags (`tag_` prefix)
`tag_title`, `tag_artist`, `tag_album`, `tag_genre`, `tag_bpm`, `tag_key`, `tag_comment`, `tag_copyright`, `tag_publisher`, `tag_isrc`, `tag_mood`

### 🤖 AI Classification (`ai_` prefix)
`ai_top_class`, `ai_top_score`, `ai_top5_classes`, `ai_top5_scores`, `ai_classifications`, `ai_frame_class_distribution`, `ai_frame_count`

### 🔗 Google Drive Links
`drive_file_id`, `drive_preview_url`, `drive_download_url`, `drive_stream_url`, `drive_embed_url`

### 🔎 Heuristics
`heuristic_is_percussive`, `heuristic_is_tonal`, `heuristic_is_noise`, `heuristic_is_bright`, `heuristic_is_dark`, `heuristic_is_loopable`, `heuristic_is_one_shot`, `heuristic_is_long_form`, `heuristic_sound_type`

---

## Supported Audio Formats
`.wav` `.mp3` `.flac` `.ogg` `.aiff` `.aif` `.m4a` `.opus` `.wma` `.caf` `.aac` `.ac3` `.amr` `.au`

---

## Resuming an Interrupted Run
If the Colab session disconnects mid-run, just **re-run all cells** — the checkpoint system will skip already-processed files.

---

## Future: Web App
The `sounds.json` output is designed to directly feed a web-based **advanced sound search** interface with:
- Full-text search across filename, tags, AI classes, and keywords
- Filters by duration, BPM, format, channels, sample rate, sound type, and acoustic features
- Inline audio preview (Google Drive streaming)
- One-click download and Drive preview links
