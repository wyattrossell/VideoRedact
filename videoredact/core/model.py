"""Core data model: transcript, redactions, video tracks, project.

All times are seconds (float). Video boxes are stored normalized (0..1)
so a project is independent of the preview resolution.
"""
from __future__ import annotations

import bisect
import datetime as _dt
import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

PROJECT_VERSION = 1


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def as_enum(cls, v):
    """Coerce a value to enum `cls` (None stays None). Qt hands str-based enums
    back from QComboBox.itemData() as plain str, so every UI entry point and the
    serializer go through this."""
    if v is None or isinstance(v, cls):
        return v
    return cls(v)


class AudioStyle(str, Enum):
    BEEP = "beep"          # 1 kHz tone (frequency configurable)
    SILENCE = "silence"
    LOW_TONE = "low_tone"  # 400 Hz, less harsh
    NOISE = "noise"        # white noise


class VideoStyle(str, Enum):
    BLACK = "black"
    BLUR = "blur"
    PIXELATE = "pixelate"


class Shape(str, Enum):
    RECT = "rect"
    ELLIPSE = "ellipse"


# --------------------------------------------------------------------------
# Transcript
# --------------------------------------------------------------------------
@dataclass
class Word:
    start: float
    end: float
    text: str
    confidence: float = 1.0

    @staticmethod
    def from_dict(d: dict) -> "Word":
        return Word(d["start"], d["end"], d["text"], d.get("confidence", 1.0))


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "Segment":
        return Segment(d["start"], d["end"], d["text"],
                       [Word.from_dict(w) for w in d.get("words", [])])


# --------------------------------------------------------------------------
# Audio redaction
# --------------------------------------------------------------------------
@dataclass
class AudioRedaction:
    start: float
    end: float
    text: str = ""                 # words covered (for the report)
    reason: str = ""
    style: Optional[AudioStyle] = None  # None -> project default
    source: str = "manual"         # manual | word_match | range | pii
    enabled: bool = True
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=_now)
    created_by: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        d = asdict(self)
        st = as_enum(AudioStyle, self.style)
        d["style"] = st.value if st else None
        return d

    @staticmethod
    def from_dict(d: dict) -> "AudioRedaction":
        d = dict(d)
        d["style"] = AudioStyle(d["style"]) if d.get("style") else None
        return AudioRedaction(**d)


# --------------------------------------------------------------------------
# Video tracks
# --------------------------------------------------------------------------
@dataclass
class BBox:
    """Normalized box: x, y = top-left, w, h = size, all in 0..1."""
    x: float
    y: float
    w: float
    h: float

    def clamp(self) -> "BBox":
        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        w = min(max(self.w, 0.0), 1.0 - x)
        h = min(max(self.h, 0.0), 1.0 - y)
        return BBox(x, y, w, h)

    def to_pixels(self, width: int, height: int, pad: float = 0.0) -> tuple[int, int, int, int]:
        """Return integer (x0, y0, x1, y1) in pixel space, padded by `pad`
        fraction of the box size on each side and clamped to the frame."""
        px = self.w * pad
        py = self.h * pad
        x0 = int(round((self.x - px) * width))
        y0 = int(round((self.y - py) * height))
        x1 = int(round((self.x + self.w + px) * width))
        y1 = int(round((self.y + self.h + py) * height))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        return x0, y0, x1, y1

    @staticmethod
    def from_pixels(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> "BBox":
        return BBox(x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height).clamp()

    def iou(self, other: "BBox") -> float:
        ax1, ay1 = self.x + self.w, self.y + self.h
        bx1, by1 = other.x + other.w, other.y + other.h
        iw = max(0.0, min(ax1, bx1) - max(self.x, other.x))
        ih = max(0.0, min(ay1, by1) - max(self.y, other.y))
        inter = iw * ih
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0

    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    @staticmethod
    def lerp(a: "BBox", b: "BBox", t: float) -> "BBox":
        return BBox(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                    a.w + (b.w - a.w) * t, a.h + (b.h - a.h) * t)

    def to_list(self) -> list[float]:
        return [round(self.x, 5), round(self.y, 5), round(self.w, 5), round(self.h, 5)]


@dataclass
class Span:
    """A contiguous range of frames in which the object is visible.
    keyframes maps frame index -> BBox; frames between keyframes are
    linearly interpolated, frames outside the keyframes but inside the
    span hold the nearest keyframe."""
    start_frame: int
    end_frame: int   # inclusive
    keyframes: dict[int, BBox] = field(default_factory=dict)

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame

    def bbox_at(self, frame: int) -> Optional[BBox]:
        if not self.contains(frame) or not self.keyframes:
            return None
        if frame in self.keyframes:
            return self.keyframes[frame]
        keys = sorted(self.keyframes)
        i = bisect.bisect_left(keys, frame)
        if i == 0:
            return self.keyframes[keys[0]]
        if i >= len(keys):
            return self.keyframes[keys[-1]]
        k0, k1 = keys[i - 1], keys[i]
        t = (frame - k0) / (k1 - k0)
        return BBox.lerp(self.keyframes[k0], self.keyframes[k1], t)

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "keyframes": {str(k): v.to_list() for k, v in sorted(self.keyframes.items())},
        }

    @staticmethod
    def from_dict(d: dict) -> "Span":
        return Span(int(d["start_frame"]), int(d["end_frame"]),
                    {int(k): BBox(*v) for k, v in d.get("keyframes", {}).items()})

    def simplify(self, tolerance: float = 0.004) -> None:
        """Drop keyframes that are well predicted by linear interpolation of
        their neighbours (keeps project files small for dense tracks)."""
        keys = sorted(self.keyframes)
        if len(keys) < 3:
            return
        keep = [keys[0]]
        last_kept = keys[0]
        for i in range(1, len(keys) - 1):
            k = keys[i]
            nxt = keys[i + 1]
            a, b = self.keyframes[last_kept], self.keyframes[nxt]
            t = (k - last_kept) / (nxt - last_kept)
            pred = BBox.lerp(a, b, t)
            real = self.keyframes[k]
            err = max(abs(pred.x - real.x), abs(pred.y - real.y),
                      abs(pred.w - real.w), abs(pred.h - real.h))
            if err > tolerance:
                keep.append(k)
                last_kept = k
        keep.append(keys[-1])
        self.keyframes = {k: self.keyframes[k] for k in keep}


@dataclass
class VideoTrack:
    label: str = "object"              # face, screen, document, person, manual...
    spans: list[Span] = field(default_factory=list)
    style: Optional[VideoStyle] = None  # None -> project default
    shape: Shape = Shape.RECT
    pad: float = 0.10                  # extra margin as fraction of box size
    source: str = "manual"             # manual | tracked | auto
    enabled: bool = True
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=_now)
    created_by: str = ""
    note: str = ""

    def bbox_at(self, frame: int) -> Optional[BBox]:
        for s in self.spans:
            if s.contains(frame):
                return s.bbox_at(frame)
        return None

    def first_frame(self) -> int:
        return min((s.start_frame for s in self.spans), default=0)

    def last_frame(self) -> int:
        return max((s.end_frame for s in self.spans), default=0)

    def total_frames(self) -> int:
        return sum(s.end_frame - s.start_frame + 1 for s in self.spans)

    def add_keyframe(self, frame: int, bbox: BBox) -> None:
        """Add/replace a keyframe. If no span contains the frame, a new
        single-frame span is created (use extend_span to widen it)."""
        for s in self.spans:
            if s.contains(frame):
                s.keyframes[frame] = bbox
                return
        self.spans.append(Span(frame, frame, {frame: bbox}))
        self.spans.sort(key=lambda s: s.start_frame)

    def set_keyframe_extend(self, frame: int, bbox: BBox) -> None:
        """Set a keyframe; if the frame falls in a gap, extend the nearest span
        to reach it (used when the user edits a box on a frame the track does
        not cover yet)."""
        for s in self.spans:
            if s.contains(frame):
                s.keyframes[frame] = bbox
                return
        if not self.spans:
            self.spans.append(Span(frame, frame, {frame: bbox}))
            return
        nearest = min(self.spans, key=lambda s: min(abs(s.start_frame - frame), abs(s.end_frame - frame)))
        nearest.start_frame = min(nearest.start_frame, frame)
        nearest.end_frame = max(nearest.end_frame, frame)
        nearest.keyframes[frame] = bbox
        self.merge_spans()

    def remove_keyframe(self, frame: int) -> None:
        for s in self.spans:
            if frame in s.keyframes:
                del s.keyframes[frame]
                if not s.keyframes:
                    self.spans.remove(s)
                return

    def merge_spans(self) -> None:
        """Merge overlapping / adjacent spans."""
        self.spans.sort(key=lambda s: s.start_frame)
        merged: list[Span] = []
        for s in self.spans:
            if merged and s.start_frame <= merged[-1].end_frame + 1:
                m = merged[-1]
                m.end_frame = max(m.end_frame, s.end_frame)
                m.keyframes.update(s.keyframes)
            else:
                merged.append(s)
        self.spans = merged

    def to_dict(self) -> dict:
        st = as_enum(VideoStyle, self.style)
        return {
            "id": self.id, "label": self.label, "style": st.value if st else None,
            "shape": as_enum(Shape, self.shape).value, "pad": self.pad, "source": self.source,
            "enabled": self.enabled, "created_at": self.created_at,
            "created_by": self.created_by, "note": self.note,
            "spans": [s.to_dict() for s in self.spans],
        }

    @staticmethod
    def from_dict(d: dict) -> "VideoTrack":
        return VideoTrack(
            id=d["id"], label=d.get("label", "object"),
            style=VideoStyle(d["style"]) if d.get("style") else None,
            shape=Shape(d.get("shape", "rect")), pad=d.get("pad", 0.1),
            source=d.get("source", "manual"), enabled=d.get("enabled", True),
            created_at=d.get("created_at", _now()), created_by=d.get("created_by", ""),
            note=d.get("note", ""),
            spans=[Span.from_dict(s) for s in d.get("spans", [])],
        )


# --------------------------------------------------------------------------
# Media info + project
# --------------------------------------------------------------------------
@dataclass
class MediaInfo:
    path: str = ""
    duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    sample_rate: int = 0
    channels: int = 0
    video_codec: str = ""
    audio_codec: str = ""
    sha256: str = ""

    def time_to_frame(self, t: float) -> int:
        return int(round(t * self.fps)) if self.fps else 0

    def frame_to_time(self, f: int) -> float:
        return f / self.fps if self.fps else 0.0

    @staticmethod
    def from_dict(d: dict) -> "MediaInfo":
        return MediaInfo(**{k: v for k, v in d.items() if k in MediaInfo.__dataclass_fields__})


@dataclass
class LogEntry:
    ts: str
    user: str
    action: str
    details: str = ""


@dataclass
class Project:
    media: MediaInfo = field(default_factory=MediaInfo)
    transcript: list[Segment] = field(default_factory=list)
    transcript_model: str = ""
    audio_redactions: list[AudioRedaction] = field(default_factory=list)
    video_tracks: list[VideoTrack] = field(default_factory=list)
    default_audio_style: AudioStyle = AudioStyle.BEEP
    default_video_style: VideoStyle = VideoStyle.BLACK
    beep_frequency: float = 1000.0
    author: str = ""
    case_number: str = ""
    notes: str = ""
    created_at: str = field(default_factory=_now)
    modified_at: str = field(default_factory=_now)
    audit_log: list[LogEntry] = field(default_factory=list)
    path: Optional[str] = None  # where the project was saved (not serialized)

    # ---- audit -----------------------------------------------------------
    def log(self, action: str, details: str = "") -> None:
        self.audit_log.append(LogEntry(_now(), self.author, action, details))
        self.modified_at = _now()

    # ---- audio helpers -------------------------------------------------
    def add_audio_redaction(self, r: AudioRedaction) -> AudioRedaction:
        r.created_by = r.created_by or self.author
        self.audio_redactions.append(r)
        self.audio_redactions.sort(key=lambda x: x.start)
        self.log("audio_redaction_added", f"{r.start:.2f}-{r.end:.2f} '{r.text}' ({r.source})")
        return r

    def remove_audio_redaction(self, rid: str) -> None:
        for r in self.audio_redactions:
            if r.id == rid:
                self.audio_redactions.remove(r)
                self.log("audio_redaction_removed", f"{r.start:.2f}-{r.end:.2f} '{r.text}'")
                return

    def is_time_redacted(self, t: float) -> bool:
        return any(r.enabled and r.start <= t < r.end for r in self.audio_redactions)

    def word_is_redacted(self, w: Word) -> bool:
        mid = (w.start + w.end) / 2
        return self.is_time_redacted(mid)

    def all_words(self) -> list[Word]:
        return [w for s in self.transcript for w in s.words]

    # ---- video helpers -------------------------------------------------
    def add_video_track(self, t: VideoTrack) -> VideoTrack:
        t.created_by = t.created_by or self.author
        self.video_tracks.append(t)
        self.log("video_track_added", f"{t.label} frames {t.first_frame()}-{t.last_frame()} ({t.source})")
        return t

    def remove_video_track(self, tid: str) -> None:
        for t in self.video_tracks:
            if t.id == tid:
                self.video_tracks.remove(t)
                self.log("video_track_removed", f"{t.label} frames {t.first_frame()}-{t.last_frame()}")
                return

    def get_track(self, tid: str) -> Optional[VideoTrack]:
        return next((t for t in self.video_tracks if t.id == tid), None)

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": "videoredact-project",
            "version": PROJECT_VERSION,
            "media": asdict(self.media),
            "transcript": [asdict(s) for s in self.transcript],
            "transcript_model": self.transcript_model,
            "audio_redactions": [r.to_dict() for r in self.audio_redactions],
            "video_tracks": [t.to_dict() for t in self.video_tracks],
            "default_audio_style": as_enum(AudioStyle, self.default_audio_style).value,
            "default_video_style": as_enum(VideoStyle, self.default_video_style).value,
            "beep_frequency": self.beep_frequency,
            "author": self.author,
            "case_number": self.case_number,
            "notes": self.notes,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "audit_log": [asdict(e) for e in self.audit_log],
        }

    @staticmethod
    def from_dict(d: dict) -> "Project":
        if d.get("format") != "videoredact-project":
            raise ValueError("Not a VideoRedact project file")
        return Project(
            media=MediaInfo.from_dict(d.get("media", {})),
            transcript=[Segment.from_dict(s) for s in d.get("transcript", [])],
            transcript_model=d.get("transcript_model", ""),
            audio_redactions=[AudioRedaction.from_dict(r) for r in d.get("audio_redactions", [])],
            video_tracks=[VideoTrack.from_dict(t) for t in d.get("video_tracks", [])],
            default_audio_style=AudioStyle(d.get("default_audio_style", "beep")),
            default_video_style=VideoStyle(d.get("default_video_style", "black")),
            beep_frequency=d.get("beep_frequency", 1000.0),
            author=d.get("author", ""), case_number=d.get("case_number", ""),
            notes=d.get("notes", ""), created_at=d.get("created_at", _now()),
            modified_at=d.get("modified_at", _now()),
            audit_log=[LogEntry(**e) for e in d.get("audit_log", [])],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        for t in self.video_tracks:
            for s in t.spans:
                s.simplify()
        self.modified_at = _now()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        tmp.replace(path)
        self.path = str(path)

    @staticmethod
    def load(path: str | Path) -> "Project":
        p = Project.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        p.path = str(path)
        return p
