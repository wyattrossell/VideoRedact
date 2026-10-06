"""Speech-to-text with word timestamps using faster-whisper (CTranslate2, CPU int8).

Models are stored under the models directory (see videoredact.paths) so the
installer can ship them and machines without internet still work.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, Iterable, Optional

import numpy as np

from videoredact.core.model import Segment, Word
from videoredact.paths import whisper_dir

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3-turbo"]
DEFAULT_MODEL = "small"

ProgressCB = Callable[[float, str], None]


def normalize_token(s: str) -> str:
    """Lower-case, strip accents and punctuation: used for word matching."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^\w']+", "", s)
    return s.strip("'")


class Transcriber:
    def __init__(self, model_size: str = DEFAULT_MODEL, device: str = "cpu",
                 compute_type: str = "int8", cpu_threads: int = 0, local_only: bool = False):
        from faster_whisper import WhisperModel
        self.model_size = model_size
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                  cpu_threads=cpu_threads, download_root=str(whisper_dir()),
                                  local_files_only=local_only)

    def transcribe(self, audio: np.ndarray | str, duration: float = 0.0, language: Optional[str] = None,
                   progress: Optional[ProgressCB] = None, cancel: Optional[Callable[[], bool]] = None,
                   vad: bool = True) -> list[Segment]:
        """audio: mono float32 at 16 kHz, or a file path."""
        segments, info = self.model.transcribe(
            audio, language=language, word_timestamps=True, vad_filter=vad,
            vad_parameters=dict(min_silence_duration_ms=400),
            beam_size=5, condition_on_previous_text=False)
        total = duration or (info.duration if info and info.duration else 0.0)
        out: list[Segment] = []
        for seg in segments:
            if cancel and cancel():
                break
            words = [Word(float(w.start), float(w.end), w.word.strip(), float(w.probability))
                     for w in (seg.words or []) if w.word.strip()]
            if not words:
                words = [Word(float(seg.start), float(seg.end), seg.text.strip(), 1.0)]
            out.append(Segment(float(seg.start), float(seg.end), seg.text.strip(), words))
            if progress and total:
                progress(min(0.999, seg.end / total), f"Transcribing {seg.end:0.0f}s / {total:0.0f}s")
        if progress:
            progress(1.0, "Transcription complete")
        return out


# --------------------------------------------------------------------------
# Word / phrase matching across a transcript
# --------------------------------------------------------------------------
def find_matches(segments: Iterable[Segment], phrase: str) -> list[list[Word]]:
    """Return every occurrence of `phrase` (one or more words) as the list of
    Word objects that make it up. Matching is case/punctuation-insensitive."""
    target = [normalize_token(t) for t in phrase.split()]
    target = [t for t in target if t]
    if not target:
        return []
    words = [w for s in segments for w in s.words]
    norm = [normalize_token(w.text) for w in words]
    n = len(target)
    hits: list[list[Word]] = []
    for i in range(len(words) - n + 1):
        if norm[i:i + n] == target:
            hits.append(words[i:i + n])
    return hits


def words_in_range(segments: Iterable[Segment], start: float, end: float) -> list[Word]:
    return [w for s in segments for w in s.words if w.end > start and w.start < end]
