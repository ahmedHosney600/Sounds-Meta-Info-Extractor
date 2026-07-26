"""extractors/heuristics.py — Derived classification flags from extracted features."""

from __future__ import annotations


def compute_heuristics(
    lb: dict,
    ai: dict,
    fn_parsed: dict,
) -> dict:
    """
    Compute heuristic flags and a coarse sound-type label from extracted features.

    Parameters
    ----------
    lb        : Dict returned by get_librosa_features().
    ai        : Dict returned by YAMNetClassifier.classify().
    fn_parsed : Dict returned by parse_filename().

    Flags returned
    --------------
    heuristic_is_percussive  : High ZCR / percussive ratio → likely drum/hit
    heuristic_is_tonal       : High harmonic ratio + low flatness → pitched/musical
    heuristic_is_noise       : High flatness → noise or texture
    heuristic_is_bright      : Spectral centroid > 4 kHz
    heuristic_is_dark        : Spectral centroid < 1.2 kHz
    heuristic_is_loopable    : Longer than 1s + detected tempo + loop keyword or duration > 4s
    heuristic_is_one_shot    : Duration < 4s or one-shot keyword found
    heuristic_is_long_form   : Duration > 30s (ambience, music, etc.)
    heuristic_sound_type     : Coarse category string derived from YAMNet classes
    """
    dur        = lb.get("lb_duration_seconds", 0) or 0
    zcr        = lb.get("lb_zcr_mean", 0) or 0
    perc       = lb.get("lb_percussive_ratio", 0) or 0
    harm       = lb.get("lb_harmonic_ratio", 0) or 0
    onset_rate = lb.get("lb_onset_rate_per_sec", 0) or 0
    tempo      = lb.get("lb_tempo_bpm", 0) or 0
    flatness   = lb.get("lb_spectral_flatness_mean", 0) or 0
    centroid   = lb.get("lb_spectral_centroid_mean", 0) or 0

    r: dict = {
        "heuristic_is_percussive": bool(perc > 0.5 or (zcr > 0.08 and onset_rate > 2)),
        "heuristic_is_tonal":      bool(harm > 0.55 and flatness < 0.02),
        "heuristic_is_noise":      bool(flatness > 0.06),
        "heuristic_is_bright":     bool(centroid > 4000),
        "heuristic_is_dark":       bool(0 < centroid < 1200),
        "heuristic_is_loopable":   bool(
            dur > 1.0 and tempo > 40
            and (fn_parsed.get("flag_has_loop") or dur > 4.0)
        ),
        "heuristic_is_one_shot":   bool(dur < 4.0 or fn_parsed.get("flag_has_oneshot")),
        "heuristic_is_long_form":  bool(dur > 30),
    }

    # ── Coarse sound type from YAMNet top-5 classes ──────────────────────────
    ai_top5 = " ".join(c.lower() for c in (ai.get("ai_top5_classes") or []))

    def _m(*keywords: str) -> bool:
        return any(k in ai_top5 for k in keywords)

    if _m("drum", "percussion", "clap", "snare", "kick", "hi-hat", "cymbal", "rimshot"):
        sound_type = "Percussion"
    elif _m("wind", "air", "blow", "whoosh", "breeze"):
        sound_type = "Wind/Air"
    elif _m("explosion", "bang", "gunshot", "cannon", "impact", "thud", "slam"):
        sound_type = "Impact/Explosion"
    elif _m("music", "singing", "melody", "chord", "piano", "guitar", "violin", "synth"):
        sound_type = "Musical"
    elif _m("speech", "voice", "talk", "shout", "whisper", "crowd", "cheer"):
        sound_type = "Voice/Speech"
    elif _m("nature", "rain", "water", "bird", "animal", "thunder", "ocean", "river", "forest"):
        sound_type = "Nature/Ambience"
    elif _m("footstep", "walking", "door", "creak", "knock", "cloth", "rustle", "scratch"):
        sound_type = "Foley"
    elif _m("engine", "motor", "vehicle", "car", "machine", "mechanical", "gear"):
        sound_type = "Mechanical"
    elif r["heuristic_is_noise"]:
        sound_type = "Noise/Texture"
    elif r["heuristic_is_tonal"]:
        sound_type = "Tonal/Instrument"
    else:
        sound_type = "Other"

    r["heuristic_sound_type"] = sound_type
    return r
