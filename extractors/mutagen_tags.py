"""extractors/mutagen_tags.py — Read embedded metadata tags from audio files."""

from mutagen import File as MutagenFile


def _first(val) -> str | None:
    """Return the first string value from a mutagen tag, or None."""
    if val is None:
        return None
    if isinstance(val, list) and val:
        val = val[0]
    s = str(val).strip()
    return s if s else None


def get_embedded_tags(filepath: str) -> dict:
    """
    Read embedded metadata tags from an audio file.

    Format support (via mutagen)
    ----------------------------
    - MP3    : ID3v1 / ID3v2
    - FLAC   : Vorbis Comment
    - OGG    : Vorbis Comment
    - WAV    : RIFF INFO + ID3
    - AIFF   : ID3
    - M4A    : iTunes MP4 atoms
    - WMA    : ASF tags
    - OPUS   : Vorbis Comment

    Uses ``easy=True`` mode for a unified key namespace across formats,
    then falls back to raw tag keys for less common fields.

    Fields returned
    ---------------
    tag_title, tag_artist, tag_album, tag_album_artist, tag_genre,
    tag_year, tag_track_number, tag_disc_number, tag_bpm, tag_key,
    tag_comment, tag_copyright, tag_publisher, tag_isrc, tag_composer,
    tag_encoder, tag_description, tag_software, tag_mood, tag_occasion,
    tag_has_any_tags (bool), tag_raw_keys (list)
    """
    r: dict = {
        "tag_title":        None,
        "tag_artist":       None,
        "tag_album":        None,
        "tag_album_artist": None,
        "tag_genre":        None,
        "tag_year":         None,
        "tag_track_number": None,
        "tag_disc_number":  None,
        "tag_bpm":          None,
        "tag_key":          None,
        "tag_comment":      None,
        "tag_copyright":    None,
        "tag_publisher":    None,
        "tag_isrc":         None,
        "tag_composer":     None,
        "tag_encoder":      None,
        "tag_description":  None,
        "tag_software":     None,
        "tag_mood":         None,
        "tag_occasion":     None,
        "tag_has_any_tags": False,
        "tag_raw_keys":     [],
    }

    try:
        audio = MutagenFile(filepath, easy=True)
        if audio is None:
            return r

        keys = list(audio.keys())
        r["tag_raw_keys"]     = keys
        r["tag_has_any_tags"] = len(keys) > 0

        def g(*tag_keys: str) -> str | None:
            for k in tag_keys:
                v = _first(audio.get(k))
                if v:
                    return v
            return None

        r["tag_title"]        = g("title",       "TIT2",  "\xa9nam")
        r["tag_artist"]       = g("artist",      "TPE1",  "\xa9ART", "author")
        r["tag_album"]        = g("album",        "TALB",  "\xa9alb")
        r["tag_album_artist"] = g("albumartist", "TPE2",  "aART")
        r["tag_genre"]        = g("genre",        "TCON",  "\xa9gen")
        r["tag_year"]         = g("date",         "TDRC",  "TYER",  "\xa9day", "year")
        r["tag_track_number"] = g("tracknumber",  "TRCK",  "trkn")
        r["tag_disc_number"]  = g("discnumber",   "TPOS")
        r["tag_bpm"]          = g("bpm",          "TBPM",  "tmpo",  "beatsperminute")
        r["tag_key"]          = g("key",          "TKEY",  "initialkey")
        r["tag_comment"]      = g("comment",      "COMM",  "\xa9cmt", "description")
        r["tag_copyright"]    = g("copyright",    "TCOP",  "cprt")
        r["tag_publisher"]    = g("organization", "TPUB",  "\xa9pub", "label")
        r["tag_composer"]     = g("composer",     "TCOM",  "\xa9wrt")
        r["tag_encoder"]      = g("encodedby",    "TENC",  "\xa9too", "encoded-by")
        r["tag_isrc"]         = g("isrc",         "TSRC")
        r["tag_description"]  = g("TIT3",         "subtitle", "contentdescription")
        r["tag_software"]     = g("ISFT",         "software")
        r["tag_mood"]         = g("mood",         "TMOO")
        r["tag_occasion"]     = g("occasion",     "contentgroup", "TIT1")

    except Exception as exc:
        r["tag_error"] = str(exc)

    return r
