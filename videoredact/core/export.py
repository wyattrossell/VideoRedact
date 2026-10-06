"""Export pipeline: apply all redactions and write the output media.

Video: frames are read sequentially with OpenCV, redacted in Python, and
piped as raw BGR into FFmpeg which encodes H.264 (libx264) and muxes the
redacted audio. Audio-only files: the redacted PCM is encoded directly.

A sidecar report (CSV + PDF) and the project file are written next to the
output when requested. The original file is never modified.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import audio_redact, video_redact
from .media import FrameReader, decode_audio, ffmpeg_exe, probe, sha256_file, write_wav
from .model import Project

ProgressCB = Callable[[float, str], None]
CancelCB = Callable[[], bool]
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class ExportOptions:
    output_path: str
    crf: int = 18
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    audio_pad_s: float = 0.05
    write_report: bool = True
    write_project: bool = True
    threads: int = 0  # 0 = ffmpeg default (all cores)
    verify_labels: Optional[list] = None   # e.g. ["face"]: re-detect in the output and report uncovered hits


class ExportCancelled(Exception):
    pass


def _report_paths(out: Path) -> tuple[Path, Path, Path]:
    stem = out.with_suffix("")
    return (Path(str(stem) + "_redaction_report.csv"),
            Path(str(stem) + "_redaction_report.pdf"),
            Path(str(stem) + ".vrproj"))


def export_project(project: Project, opts: ExportOptions,
                   progress: Optional[ProgressCB] = None,
                   cancel: Optional[CancelCB] = None) -> dict:
    """Run the export. Returns a dict with output paths and hashes."""
    def prog(p: float, msg: str) -> None:
        if progress:
            progress(p, msg)

    def check_cancel() -> None:
        if cancel and cancel():
            raise ExportCancelled()

    src = project.media.path
    info = project.media if project.media.frame_count or project.media.has_audio else probe(src)
    out_path = Path(opts.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tmpdir = Path(tempfile.mkdtemp(prefix="videoredact_"))
    wav_path = tmpdir / "audio.wav"
    result: dict = {"output": str(out_path)}

    try:
        # ---------------- audio ----------------
        has_audio = info.has_audio
        if has_audio:
            prog(0.01, "Decoding audio")
            samples, sr = decode_audio(src)
            check_cancel()
            prog(0.04, "Applying audio redactions")
            samples = audio_redact.apply_redactions(
                samples, sr, project.audio_redactions, project.default_audio_style,
                project.beep_frequency, pad_s=opts.audio_pad_s)
            write_wav(wav_path, samples, sr)
            del samples
            check_cancel()

        # ---------------- video ----------------
        if info.has_video:
            prog(0.06, "Encoding video")
            with FrameReader(src) as reader:
                W, H = reader.width, reader.height
                fps = info.fps or reader.fps
                total = reader.frame_count or info.frame_count or 1
                cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:.6f}",
                       "-i", "pipe:0"]
                if has_audio:
                    cmd += ["-i", str(wav_path)]
                cmd += ["-map", "0:v:0"]
                if has_audio:
                    cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", opts.audio_bitrate]
                cmd += ["-c:v", "libx264", "-preset", opts.preset, "-crf", str(opts.crf),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
                if opts.threads:
                    cmd += ["-threads", str(opts.threads)]
                if has_audio:
                    cmd += ["-shortest"]
                cmd += [str(out_path)]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                        creationflags=_CREATE_NO_WINDOW)
                tracks = [t for t in project.video_tracks if t.enabled]
                last_prog = 0.0
                try:
                    for idx, frame in reader.iter_frames():
                        if tracks:
                            video_redact.redact_frame(frame, tracks, idx, project.default_video_style)
                        proc.stdin.write(frame.tobytes())
                        if idx % 15 == 0:
                            p = 0.06 + 0.90 * min(1.0, idx / total)
                            if p - last_prog > 0.002:
                                prog(p, f"Encoding frame {idx}/{total}")
                                last_prog = p
                            if cancel and cancel():
                                proc.kill()
                                raise ExportCancelled()
                finally:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                err = proc.stderr.read().decode("utf-8", "replace")
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"FFmpeg encode failed (code {rc}): {err[-1500:]}")
        else:
            # audio only
            prog(0.5, "Encoding audio")
            ext = out_path.suffix.lower()
            codec = {".wav": ["-c:a", "pcm_s16le"], ".mp3": ["-c:a", "libmp3lame", "-b:a", opts.audio_bitrate],
                     ".flac": ["-c:a", "flac"], ".ogg": ["-c:a", "libvorbis"]}.get(ext, ["-c:a", "aac", "-b:a", opts.audio_bitrate])
            res = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav_path),
                                  *codec, str(out_path)], capture_output=True, creationflags=_CREATE_NO_WINDOW)
            if res.returncode != 0:
                raise RuntimeError("FFmpeg audio encode failed: " + res.stderr.decode("utf-8", "replace")[-1500:])

        # ---------------- hashes + sidecars ----------------
        prog(0.97, "Hashing output")
        result["output_sha256"] = sha256_file(out_path)
        if not project.media.sha256:
            prog(0.98, "Hashing source")
            project.media.sha256 = sha256_file(src)
        result["source_sha256"] = project.media.sha256
        result["elapsed_s"] = time.time() - t0
        project.log("export", f"{out_path.name} sha256={result['output_sha256'][:16]}... in {result['elapsed_s']:.0f}s")

        csv_p, pdf_p, proj_p = _report_paths(out_path)
        if opts.write_project:
            project.save(proj_p)
            result["project"] = str(proj_p)
        if opts.write_report:
            prog(0.99, "Writing report")
            from .report import write_csv, write_pdf
            write_csv(project, csv_p, result)
            write_pdf(project, pdf_p, result)
            result["report_csv"] = str(csv_p)
            result["report_pdf"] = str(pdf_p)
        if opts.verify_labels:
            prog(0.99, "Verifying the redacted output…")
            from .verify import verify_output
            vr = verify_output(str(out_path), project, labels=list(opts.verify_labels),
                               progress=lambda p, m: prog(0.99 + 0.01 * p, m), cancel=cancel)
            result["verify"] = vr
            project.log("verify", f"{vr.frames_checked} frames checked, {vr.detections} detections, "
                                  f"{len(vr.uncovered)} uncovered")
            if vr.uncovered:
                vpath = Path(str(out_path.with_suffix("")) + "_verification.csv")
                with open(vpath, "w", encoding="utf-8", newline="") as f:
                    f.write("time,frame,label,score,coverage,x,y,w,h\n")
                    for u in vr.uncovered:
                        f.write(f"{u.time:.2f},{u.frame},{u.label},{u.score:.2f},{u.covered:.2f},"
                                f"{u.bbox.x:.4f},{u.bbox.y:.4f},{u.bbox.w:.4f},{u.bbox.h:.4f}\n")
                result["verification_csv"] = str(vpath)
        prog(1.0, "Export complete")
        return result
    finally:
        try:
            if wav_path.exists():
                os.remove(wav_path)
            os.rmdir(tmpdir)
        except Exception:
            pass
