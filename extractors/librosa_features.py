"""extractors/librosa_features.py — Full spectral & acoustic feature extraction."""

import warnings
import numpy as np
import librosa

import config

warnings.filterwarnings("ignore")


def get_librosa_features(
    filepath: str,
    sr: int | None = None,
    n_mfcc: int | None = None,
) -> dict:
    """
    Extract comprehensive audio features using librosa.

    Feature groups
    --------------
    Temporal   : duration, samples, peak amplitude (linear + dB)
    Energy     : RMS mean/max/std/dB, dynamic range
    ZCR        : zero-crossing rate mean + std
    Spectral   : centroid, bandwidth, rolloff, flatness, contrast (7 bands)
    Mel        : mel-spectrogram mean/max/min/std in dB
    HPSS       : harmonic ratio, percussive ratio
    Rhythm     : tempo (BPM), beat count
    Onsets     : count, rate per second, timestamp list
    Chroma     : 12-note energy vectors + dominant note
    Tonnetz    : 6-dim tonal centroid
    MFCCs      : mean + std for each of N_MFCC coefficients (also flat columns)
    Poly       : linear spectral trend coefficients

    Parameters
    ----------
    filepath : Path to the audio file.
    sr       : Analysis sample rate (default: config.LIBROSA_SR).
    n_mfcc   : Number of MFCCs (default: config.N_MFCC).
    """
    if sr is None:
        sr = config.LIBROSA_SR
    if n_mfcc is None:
        n_mfcc = config.N_MFCC

    r: dict = {}
    try:
        # librosa.load uses soundfile + audioread (ffmpeg) — handles all formats
        y, loaded_sr = librosa.load(filepath, sr=sr, mono=True, res_type="kaiser_fast")
        dur = librosa.get_duration(y=y, sr=loaded_sr)

        r["lb_loaded_sr"]         = int(loaded_sr)
        r["lb_duration_seconds"]  = round(float(dur), 6)
        r["lb_num_samples"]       = int(len(y))

        # ── Peak amplitude ────────────────────────────────────────────────────
        peak = float(np.max(np.abs(y)))
        r["lb_peak_amplitude"]    = round(peak, 8)
        r["lb_peak_amplitude_db"] = round(float(20 * np.log10(peak + 1e-10)), 4)

        # ── RMS Energy ────────────────────────────────────────────────────────
        rms = librosa.feature.rms(y=y)[0]
        r["lb_rms_mean"]          = round(float(np.mean(rms)), 8)
        r["lb_rms_max"]           = round(float(np.max(rms)), 8)
        r["lb_rms_std"]           = round(float(np.std(rms)), 8)
        r["lb_rms_db_mean"]       = round(
            float(librosa.amplitude_to_db(np.array([np.mean(rms)]))[0]), 4
        )
        r["lb_dynamic_range_db"]  = round(
            float(20 * np.log10((np.max(rms) + 1e-10) / (np.min(rms) + 1e-10))), 4
        )

        # ── Zero Crossing Rate ────────────────────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        r["lb_zcr_mean"] = round(float(np.mean(zcr)), 8)
        r["lb_zcr_std"]  = round(float(np.std(zcr)), 8)

        # ── Spectral Centroid (brightness) ────────────────────────────────────
        sc = librosa.feature.spectral_centroid(y=y, sr=loaded_sr)[0]
        r["lb_spectral_centroid_mean"] = round(float(np.mean(sc)), 4)
        r["lb_spectral_centroid_std"]  = round(float(np.std(sc)), 4)

        # ── Spectral Bandwidth ────────────────────────────────────────────────
        bw = librosa.feature.spectral_bandwidth(y=y, sr=loaded_sr)[0]
        r["lb_spectral_bandwidth_mean"] = round(float(np.mean(bw)), 4)
        r["lb_spectral_bandwidth_std"]  = round(float(np.std(bw)), 4)

        # ── Spectral Rolloff ──────────────────────────────────────────────────
        ro = librosa.feature.spectral_rolloff(y=y, sr=loaded_sr)[0]
        r["lb_spectral_rolloff_mean"] = round(float(np.mean(ro)), 4)
        r["lb_spectral_rolloff_std"]  = round(float(np.std(ro)), 4)

        # ── Spectral Flatness (0=tonal, 1=noise-like) ─────────────────────────
        fl = librosa.feature.spectral_flatness(y=y)[0]
        r["lb_spectral_flatness_mean"] = round(float(np.mean(fl)), 8)
        r["lb_spectral_flatness_std"]  = round(float(np.std(fl)), 8)

        # ── Spectral Contrast (7 frequency bands) ─────────────────────────────
        ct = librosa.feature.spectral_contrast(y=y, sr=loaded_sr)
        r["lb_spectral_contrast_mean"] = [round(float(v), 4) for v in np.mean(ct, axis=1)]

        # ── Mel Spectrogram statistics ─────────────────────────────────────────
        mel    = librosa.feature.melspectrogram(y=y, sr=loaded_sr)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        r["lb_mel_mean_db"] = round(float(np.mean(mel_db)), 4)
        r["lb_mel_max_db"]  = round(float(np.max(mel_db)), 4)
        r["lb_mel_min_db"]  = round(float(np.min(mel_db)), 4)
        r["lb_mel_std_db"]  = round(float(np.std(mel_db)), 4)

        # ── Harmonic / Percussive Source Separation (HPSS) ────────────────────
        y_h, y_p  = librosa.effects.hpss(y)
        total_e   = np.mean(np.abs(y)) + 1e-10
        r["lb_harmonic_ratio"]   = round(float(np.mean(np.abs(y_h)) / total_e), 6)
        r["lb_percussive_ratio"] = round(float(np.mean(np.abs(y_p)) / total_e), 6)

        # ── Tempo & Beat Tracking ──────────────────────────────────────────────
        tempo, beats = librosa.beat.beat_track(y=y, sr=loaded_sr)
        r["lb_tempo_bpm"]  = round(float(tempo), 4)
        r["lb_beat_count"] = int(len(beats))

        # ── Onset Detection ────────────────────────────────────────────────────
        onsets = librosa.onset.onset_detect(y=y, sr=loaded_sr, units="time")
        r["lb_onset_count"]         = int(len(onsets))
        r["lb_onset_rate_per_sec"]  = round(float(len(onsets) / dur) if dur > 0 else 0, 4)
        r["lb_onset_times_seconds"] = [round(float(t), 4) for t in onsets.tolist()]

        # ── Chroma (12-class pitch energy) ─────────────────────────────────────
        chroma   = librosa.feature.chroma_stft(y=y, sr=loaded_sr)
        chroma_m = np.mean(chroma, axis=1)
        NOTES    = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        r["lb_chroma_mean"]          = [round(float(v), 6) for v in chroma_m]
        r["lb_chroma_std"]           = [round(float(v), 6) for v in np.std(chroma, axis=1)]
        r["lb_chroma_dominant_note"] = NOTES[int(np.argmax(chroma_m))]
        r["lb_chroma_second_note"]   = NOTES[int(np.argsort(chroma_m)[-2])]

        # ── Tonnetz (tonal centroid features, 6-dim) ───────────────────────────
        try:
            tn = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=loaded_sr)
            r["lb_tonnetz_mean"] = [round(float(v), 6) for v in np.mean(tn, axis=1)]
        except Exception:
            r["lb_tonnetz_mean"] = None

        # ── MFCCs ──────────────────────────────────────────────────────────────
        mfccs  = librosa.feature.mfcc(y=y, sr=loaded_sr, n_mfcc=n_mfcc)
        mfcc_m = np.mean(mfccs, axis=1)
        mfcc_s = np.std(mfccs, axis=1)
        r["lb_mfcc_mean"] = [round(float(v), 4) for v in mfcc_m]
        r["lb_mfcc_std"]  = [round(float(v), 4) for v in mfcc_s]
        # Flat individual columns (easier for CSV-based filtering in the web app)
        for i in range(n_mfcc):
            r[f"lb_mfcc_{i + 1:02d}_mean"] = round(float(mfcc_m[i]), 4)
            r[f"lb_mfcc_{i + 1:02d}_std"]  = round(float(mfcc_s[i]), 4)

        # ── Polynomial spectral features (linear trend) ────────────────────────
        try:
            poly = librosa.feature.poly_features(y=y, sr=loaded_sr, order=1)
            r["lb_poly_coeff_0_mean"] = round(float(np.mean(poly[0])), 6)
            r["lb_poly_coeff_1_mean"] = round(float(np.mean(poly[1])), 6)
        except Exception:
            pass

    except Exception as exc:
        r["lb_error"] = str(exc)

    return r
