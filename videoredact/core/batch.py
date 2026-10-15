"""Batch processing: run auto-detect (and optionally export) over many files.

Files are processed one after another; each file already uses the parallel
detection workers, so running files concurrently would only fight for the
same cores. For every input it writes `<name>.vrproj` next to the source (or
into `out_dir`) and, when `export` is set, `<name>_REDACTED.mp4` with the
report, verification and project sidecars. A JSON summary is written at the
end so an overnight run can be reviewed in the morning.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from .export import ExportOptions, export_project
from .media import FrameReader, probe, sha256_file
from .model import AudioStyle, Project, VideoStyle

ProgressCB = Callable[[float, str], None]
CancelCB = Callable[[], bool]


@dataclass
class BatchOptions:
    files: list[str]
    out_dir: Optional[str] = None            # None -> next to each source
    detect_labels: list[str] = field(default_factory=lambda: ["face", "screen", "document"])
    stride: int = 5
    export: bool = False
    verify: bool = True
    author: str = ""
    default_video_style: str = "black"
    default_audio_style: str = "beep"
    crf: int = 23
    preset: str = "veryfast"
    detect_kwargs: dict = field(default_factory=dict)   # extra AutoDetectOptions fields


@dataclass
class BatchItemResult:
    file: str
    ok: bool
    project: str = ""
    output: str = ""
    tracks: int = 0
    seconds: float = 0.0
    uncovered: Optional[int] = None
    error: str = ""


def run_batch(opts: BatchOptions, progress: Optional[ProgressCB] = None, cancel: Optional[CancelCB] = None) -> list[BatchItemResult]:
    from videoredact.vision.auto_detect import AutoDetectOptions, run_auto_detect
    results: list[BatchItemResult] = []
    n = len(opts.files)
    for i, f in enumerate(opts.files):
        if cancel and cancel():
            break
        base = (i / n)
        span = 1 / n

        def prog(p, m, base=base, span=span, name=Path(f).name):
            if progress:
                progress(base + span * p, f"[{i + 1}/{n}] {name}: {m}")

        t0 = time.time()
        res = BatchItemResult(file=f, ok=False)
        try:
            info = probe(f)
            proj = Project(media=info, author=opts.author,
                           default_video_style=VideoStyle(opts.default_video_style),
                           default_audio_style=AudioStyle(opts.default_audio_style))
            proj.log("batch_started", f)
            prog(0.0, "hashing source")
            info.sha256 = sha256_file(f)
            if info.has_video:
                dopts = AutoDetectOptions(labels=list(opts.detect_labels), stride=opts.stride, **opts.detect_kwargs)
                with FrameReader(f) as r:
                    tracks = run_auto_detect(r, dopts, progress=lambda p, m: prog(0.05 + 0.6 * p, m),
                                             cancel=cancel)
                for t in tracks:
                    proj.add_video_track(t)
                res.tracks = len(tracks)
            out_dir = Path(opts.out_dir) if opts.out_dir else Path(f).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            proj_path = out_dir / (Path(f).stem + ".vrproj")
            proj.save(proj_path)
            res.project = str(proj_path)
            if opts.export and not (cancel and cancel()):
                out = out_dir / (Path(f).stem + "_REDACTED" + (".mp4" if info.has_video else ".m4a"))
                eo = ExportOptions(str(out), crf=opts.crf, preset=opts.preset, write_report=True, write_project=True,
                                   verify_labels=["face"] if (opts.verify and info.has_video) else None)
                r = export_project(proj, eo, progress=lambda p, m: prog(0.65 + 0.35 * p, m), cancel=cancel)
                res.output = r["output"]
                vr = r.get("verify")
                res.uncovered = len(vr.uncovered) if vr else None
            res.ok = True
        except Exception as e:  # noqa: BLE001
            res.error = f"{e}\n{traceback.format_exc()[-800:]}"
        res.seconds = time.time() - t0
        results.append(res)
    summary_dir = Path(opts.out_dir) if opts.out_dir else (Path(opts.files[0]).parent if opts.files else Path("."))
    try:
        (summary_dir / f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
            json.dumps([asdict(r) for r in results], indent=1), encoding="utf-8")
    except Exception:
        pass
    if progress:
        progress(1.0, f"Batch complete: {sum(r.ok for r in results)}/{len(results)} files")
    return results
