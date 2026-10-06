"""Speech-to-text with word timestamps using faster-whisper (CTranslate2, CPU int8).

Models are stored under the models directory (see videoredact.paths) so the
installer can ship them and machines without internet still work.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Callable, Iterable, Optional

import numpy as np

from videoredact.core.model import Segment, Word
from videoredact.paths import whisper_dir

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3-turbo"]
DEFAULT_MODEL = "small"
log = logging.getLogger(__name__)
# Whisper hallucinates short filler on silence/noise ("you", "Thank you.", "Bye.").
# Drop segments that are both high no-speech probability and generic filler.
FILLER = {"you", "thank you", "thanks", "bye", "okay", "yeah", "the", "oh", "um", "uh", "hmm", "mm"}
NO_SPEECH_DROP = 0.85

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
        root = whisper_dir(model_size)
        # If the model is already on disk (bundled or previously downloaded) never touch the
        # network: on locked-down machines the Hugging Face check can hang for a long time.
        if (root / f"models--Systran--faster-whisper-{model_size}" / "snapshots").exists():
            local_only = True
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type,
                                  cpu_threads=cpu_threads, download_root=str(root),
                                  local_files_only=local_only)

    def transcribe(self, audio: np.ndarray | str, duration: float = 0.0, language: Optional[str] = None,
                   progress: Optional[ProgressCB] = None, cancel: Optional[Callable[[], bool]] = None,
                   vad: bool = False, on_segment: Optional[Callable[[Segment], None]] = None) -> list[Segment]:
        """audio: mono float32 at 16 kHz, or a file path.
        vad=False by default: on noisy body-cam audio the Silero VAD discarded most
        real speech (12 of 14 minutes on a traffic-stop clip). Hallucinated filler is
        filtered instead. on_segment is called for each segment as it is produced so the
        UI can show results progressively."""
        segments, info = self.model.transcribe(
            audio, language=language, word_timestamps=True, vad_filter=vad,
            vad_parameters=dict(threshold=0.3, min_silence_duration_ms=1000, speech_pad_ms=400),
            beam_size=5, condition_on_previous_text=False, no_speech_threshold=0.7,
            log_prob_threshold=-1.0, compression_ratio_threshold=2.4)
        total = duration or (info.duration if info and info.duration else 0.0)
        out: list[Segment] = []
        dropped = 0
        for seg in segments:
            if cancel and cancel():
                break
            text = seg.text.strip()
            norm = normalize_token(text.replace(" ", "")) if len(text.split()) <= 2 else ""
            if not text or (seg.no_speech_prob >= NO_SPEECH_DROP and (norm in FILLER or len(text) <= 3)):
                dropped += 1
                continue
            words = [Word(float(w.start), float(w.end), w.word.strip(), float(w.probability))
                     for w in (seg.words or []) if w.word.strip()]
            if not words:
                words = [Word(float(seg.start), float(seg.end), text, 1.0)]
            s = Segment(float(seg.start), float(seg.end), text, words)
            out.append(s)
            if on_segment:
                on_segment(s)
            if progress and total:
                progress(min(0.999, seg.end / total), f"Transcribing {_fmt(seg.end)} / {_fmt(total)}  ({len(out)} segments)")
        log.info("transcribed %d segments (%d dropped as noise) model=%s vad=%s",
                 len(out), dropped, self.model_size, vad)
        if progress:
            progress(1.0, "Transcription complete")
        return out


def _fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


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
