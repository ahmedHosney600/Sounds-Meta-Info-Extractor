"""extractors/audio_technical.py — Audio technical specs via soundfile."""

import soundfile as sf

# Map soundfile subtype strings to bit depth integers
_BIT_DEPTH_MAP: dict[str, int | None] = {
    "PCM_S8": 8,  "PCM_U8": 8,
    "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
    "FLOAT":  32, "DOUBLE": 64,
    "VORBIS": None, "OPUS": None,
}


def get_audio_technical(filepath: str) -> dict:
    """
    Extract low-level audio technical metadata using soundfile.

    Natively handles WAV, FLAC, OGG, AIFF.
    MP3 / M4A / other formats are decoded via libsndfile's ffmpeg backend
    (available on Colab after ``apt-get install ffmpeg``).

    Fields returned
    ---------------
    sf_sample_rate, sf_channels, sf_frames, sf_duration_seconds,
    sf_duration_formatted (MM:SS.mmm), sf_format, sf_subtype,
    sf_bit_depth, sf_channel_label (mono/stereo/Nch),
    sf_endian, sf_sections, sf_seekable
    """
    r: dict = {
        "sf_sample_rate":       None,
        "sf_channels":          None,
        "sf_frames":            None,
        "sf_duration_seconds":  None,
        "sf_duration_formatted": None,
        "sf_format":            None,
        "sf_subtype":           None,
        "sf_bit_depth":         None,
        "sf_channel_label":     None,
        "sf_endian":            None,
        "sf_sections":          None,
        "sf_seekable":          None,
    }
    try:
        info = sf.info(filepath)
        dur = info.duration
        r.update(
            {
                "sf_sample_rate":       info.samplerate,
                "sf_channels":          info.channels,
                "sf_frames":            info.frames,
                "sf_duration_seconds":  round(float(dur), 6),
                "sf_duration_formatted": f"{int(dur // 60):02d}:{dur % 60:06.3f}",
                "sf_format":            info.format,
                "sf_subtype":           info.subtype,
                "sf_endian":            info.endian,
                "sf_sections":          info.sections,
                "sf_seekable":          info.seekable,
                "sf_channel_label":     (
                    "mono" if info.channels == 1
                    else "stereo" if info.channels == 2
                    else f"{info.channels}ch"
                ),
            }
        )
        sub = (info.subtype or "").upper()
        for k, v in _BIT_DEPTH_MAP.items():
            if k in sub:
                r["sf_bit_depth"] = v
                break
    except Exception as exc:
        r["sf_error"] = str(exc)

    return r
