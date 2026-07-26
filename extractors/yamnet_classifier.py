"""extractors/yamnet_classifier.py — YAMNet AI audio classification (521 AudioSet classes)."""

from __future__ import annotations
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

import config


_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master"
    "/research/audioset/yamnet/yamnet_class_map.csv"
)
_CLASS_MAP_LOCAL = "/content/yamnet_class_map.csv"


class YAMNetClassifier:
    """
    Loads the YAMNet model from TensorFlow Hub and provides a classify() method.

    The model is loaded once at construction and reused for all files —
    this avoids the ~10 s overhead of reloading TF Hub artifacts per file.

    YAMNet requirements
    -------------------
    - Input  : mono waveform, 16 kHz sample rate, float32 [-1, 1]
    - Output : scores tensor [num_frames × 521], embeddings, log-mel spectrogram
    """

    def __init__(self) -> None:
        import tensorflow as tf
        import tensorflow_hub as hub
        import librosa  # imported here so the class can be defined without librosa

        self._tf      = tf
        self._librosa = librosa

        print("Loading YAMNet from TensorFlow Hub (521-class AudioSet classifier)…")
        self._model = hub.load("https://tfhub.dev/google/yamnet/1")

        # Download and parse class name list
        try:
            urllib.request.urlretrieve(_CLASS_MAP_URL, _CLASS_MAP_LOCAL)
        except Exception:
            pass  # fallback to integer class IDs if download fails

        self._class_names: list[str] = []
        try:
            with open(_CLASS_MAP_LOCAL, "r") as f:
                for line in f.readlines()[1:]:  # skip header
                    parts = line.strip().split(",")
                    name  = parts[2].strip('"') if len(parts) > 2 else str(len(self._class_names))
                    self._class_names.append(name)
        except FileNotFoundError:
            self._class_names = [str(i) for i in range(521)]

        print(f"✅  YAMNet ready — {len(self._class_names)} sound classes")

    # ------------------------------------------------------------------
    def classify(self, filepath: str, top_k: int | None = None) -> dict:
        """
        Run YAMNet on *filepath* and return classification results.

        Fields returned
        ---------------
        ai_top_class, ai_top_score,
        ai_top5_classes (list), ai_top5_scores (list),
        ai_classifications (list of {class, score} dicts),
        ai_frame_class_distribution (dict: class → frame count),
        ai_frame_count (int),
        ai_error (str or None)
        """
        if top_k is None:
            top_k = config.YAMNET_TOP_K

        r: dict = {
            "ai_top_class":               None,
            "ai_top_score":               None,
            "ai_top5_classes":            [],
            "ai_top5_scores":             [],
            "ai_classifications":         [],
            "ai_frame_class_distribution": {},
            "ai_frame_count":             None,
            "ai_error":                   None,
        }
        try:
            # Load at 16 kHz mono — YAMNet requirement
            y, _ = self._librosa.load(filepath, sr=config.YAMNET_SR, mono=True)
            waveform = self._tf.constant(y, dtype=self._tf.float32)

            # scores shape: [num_frames, 521]
            scores, _embeddings, _spectrogram = self._model(waveform)
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

            # Per-frame dominant class → temporal class distribution
            frame_tops = [self._class_names[np.argmax(s)] for s in scores.numpy()]
            class_dist = Counter(frame_tops)
            r["ai_frame_class_distribution"] = dict(class_dist.most_common(10))
            r["ai_frame_count"] = int(scores.shape[0])

        except Exception as exc:
            r["ai_error"] = str(exc)

        return r
