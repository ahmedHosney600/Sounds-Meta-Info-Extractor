"""extractors/yamnet_classifier.py — YAMNet AI audio classification (521 AudioSet classes).

Accepts either a filepath or a pre-loaded waveform (numpy array at 16 kHz)
to avoid re-reading files already loaded by the caller.
"""

from __future__ import annotations
import urllib.request
from collections import Counter

import numpy as np

import config


_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master"
    "/research/audioset/yamnet/yamnet_class_map.csv"
)
_CLASS_MAP_LOCAL = "/content/yamnet_class_map.csv"


class YAMNetClassifier:
    """
    Loads YAMNet from TensorFlow Hub once and provides a classify() method.

    The model is loaded once at construction and reused across all files.
    TensorFlow 2.x GPU inference is thread-safe — the model can be called
    concurrently from multiple threads.

    YAMNet requirements
    -------------------
    - Input  : mono waveform, 16 kHz sample rate, float32 [-1, 1]
    - Output : scores [num_frames × 521], embeddings, log-mel spectrogram
    """

    def __init__(self) -> None:
        import tensorflow as tf
        import tensorflow_hub as hub
        import librosa

        self._tf      = tf
        self._librosa = librosa

        print("Loading YAMNet from TensorFlow Hub (521-class AudioSet classifier)…")
        self._model = hub.load("https://tfhub.dev/google/yamnet/1")

        # Download class name list
        try:
            urllib.request.urlretrieve(_CLASS_MAP_URL, _CLASS_MAP_LOCAL)
        except Exception:
            pass

        self._class_names: list[str] = []
        try:
            with open(_CLASS_MAP_LOCAL, "r") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(",")
                    name  = parts[2].strip('"') if len(parts) > 2 else str(len(self._class_names))
                    self._class_names.append(name)
        except FileNotFoundError:
            self._class_names = [str(i) for i in range(521)]

        print(f"✅  YAMNet ready — {len(self._class_names)} sound classes")

    # ------------------------------------------------------------------
    def classify(
        self,
        filepath: str | None = None,
        *,
        y: np.ndarray | None = None,
        top_k: int | None = None,
    ) -> dict:
        """
        Run YAMNet and return top-k class predictions.

        Parameters
        ----------
        filepath : Path to audio file. Used only when ``y`` is not supplied.
        y        : Pre-loaded mono waveform at ``config.YAMNET_SR`` (16 kHz).
                   When provided, ``filepath`` is ignored — no file I/O occurs.
        top_k    : Number of top classes to return (default: config.YAMNET_TOP_K).

        Fields returned
        ---------------
        ai_top_class, ai_top_score,
        ai_top5_classes, ai_top5_scores,
        ai_classifications [{class, score}],
        ai_frame_class_distribution {class: frame_count},
        ai_frame_count, ai_error
        """
        if top_k is None:
            top_k = config.YAMNET_TOP_K

        r: dict = {
            "ai_top_class":                None,
            "ai_top_score":                None,
            "ai_top5_classes":             [],
            "ai_top5_scores":              [],
            "ai_classifications":          [],
            "ai_frame_class_distribution": {},
            "ai_frame_count":              None,
            "ai_error":                    None,
        }
        try:
            # ── Load audio (skipped when caller supplies waveform) ────────────
            if y is None:
                if filepath is None:
                    raise ValueError("Either filepath or y must be provided")
                y, _ = self._librosa.load(filepath, sr=config.YAMNET_SR, mono=True)

            waveform    = self._tf.constant(y, dtype=self._tf.float32)
            scores, _emb, _spec = self._model(waveform)
            mean_scores = np.mean(scores.numpy(), axis=0)
            top_idx     = np.argsort(mean_scores)[::-1][:top_k]

            r["ai_top_class"]    = self._class_names[top_idx[0]]
            r["ai_top_score"]    = round(float(mean_scores[top_idx[0]]), 6)
            r["ai_top5_classes"] = [self._class_names[i] for i in top_idx]
            r["ai_top5_scores"]  = [round(float(mean_scores[i]), 6) for i in top_idx]
            r["ai_classifications"] = [
                {"class": self._class_names[i], "score": round(float(mean_scores[i]), 6)}
                for i in top_idx
            ]

            # Per-frame temporal class distribution
            frame_tops = [self._class_names[np.argmax(s)] for s in scores.numpy()]
            r["ai_frame_class_distribution"] = dict(Counter(frame_tops).most_common(10))
            r["ai_frame_count"] = int(scores.shape[0])

        except Exception as exc:
            r["ai_error"] = str(exc)

        return r
