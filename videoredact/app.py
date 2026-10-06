"""Application entry point."""
from __future__ import annotations

import os
import sys


def _setup_logging() -> None:
    """Frozen (windowed) builds have no console: send stdout/stderr and
    uncaught exceptions to %LOCALAPPDATA%/VideoRedact/logs/videoredact.log."""
    import logging
    from logging.handlers import RotatingFileHandler
    from videoredact.paths import log_dir
    handler = RotatingFileHandler(log_dir() / "videoredact.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    if getattr(sys, "frozen", False):
        class _Stream:
            def __init__(self, level):
                self.level = level
            def write(self, msg):
                msg = msg.rstrip()
                if msg:
                    logging.getLogger("std").log(self.level, msg)
            def flush(self):
                pass
        sys.stdout = _Stream(logging.INFO)
        sys.stderr = _Stream(logging.ERROR)


def _install_excepthook() -> None:
    import logging
    import traceback

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.getLogger("crash").error(text)
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            if QApplication.instance():
                box = QMessageBox(QMessageBox.Critical, "VideoRedact error",
                                  f"An unexpected error occurred:\n{exc}\n\nDetails were written to the log file.")
                box.setDetailedText(text)
                box.exec()
        except Exception:
            pass
    sys.excepthook = hook


def selftest(media: str | None = None) -> int:
    """`VideoRedact.exe --selftest [file]`: verify the installation without the GUI.
    Writes a report to stdout (console) and the log; returns 0 when everything works."""
    import logging
    import time
    import traceback
    log = logging.getLogger("selftest")
    lines: list[str] = []
    ok_all = True

    def step(name, fn):
        nonlocal ok_all
        t0 = time.perf_counter()
        try:
            out = fn()
            lines.append(f"[ok]   {name}: {out} ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            ok_all = False
            lines.append(f"[FAIL] {name}: {e}")
            log.error("%s failed:\n%s", name, traceback.format_exc())

    from videoredact import __version__
    from videoredact.paths import models_dir, user_data_dir, resource_root
    lines.append(f"VideoRedact {__version__}  frozen={getattr(sys, 'frozen', False)}")
    lines.append(f"resources: {resource_root()}")
    lines.append(f"models:    {models_dir()}")
    lines.append(f"settings:  {user_data_dir()}")

    def _ffmpeg():
        from videoredact.core.media import ffmpeg_exe, run_ffmpeg
        r = run_ffmpeg(["-version"], text=True)
        return f"{ffmpeg_exe()} -> {r.stdout.splitlines()[0][:60]}"
    step("ffmpeg", _ffmpeg)
    step("opencv", lambda: __import__("cv2").__version__ + (" CSRT" if hasattr(__import__("cv2"), "TrackerCSRT_create") else " no-CSRT"))
    step("onnxruntime", lambda: __import__("onnxruntime").__version__)
    step("ctranslate2 / faster-whisper", lambda: __import__("ctranslate2").__version__)
    step("PySide6", lambda: __import__("PySide6").__version__)
    step("reportlab", lambda: __import__("reportlab").Version)

    def _models():
        from videoredact.vision import models as reg
        return {n: ("present" if reg.is_available(n) else "not downloaded") for n in reg.MODELS}
    step("detection models", _models)

    def _face():
        from videoredact.vision import models as reg
        if not reg.is_available("yunet"):
            return "skipped (model not downloaded)"
        import numpy as np
        from videoredact.vision.detector import FaceDetector
        FaceDetector().detect(np.zeros((360, 640, 3), dtype=np.uint8))
        return "YuNet runs"
    step("face detector", _face)

    def _yolox():
        from videoredact.vision import models as reg
        if not reg.is_available("yolox_tiny") and not reg.is_available("yolox_s"):
            return "skipped (model not downloaded)"
        import numpy as np
        from videoredact.vision.detector import ObjectDetector
        name = "yolox_tiny" if reg.is_available("yolox_tiny") else "yolox_s"
        ObjectDetector(name).detect(np.zeros((360, 640, 3), dtype=np.uint8))
        return f"{name} runs"
    step("object detector", _yolox)

    if media:
        def _probe():
            from videoredact.core.media import probe
            i = probe(media)
            return f"{i.width}x{i.height} {i.fps:.2f}fps {i.duration:.1f}s audio={i.has_audio}"
        step("probe media", _probe)

        def _export():
            import tempfile
            from videoredact.core.export import ExportOptions, export_project
            from videoredact.core.media import probe
            from videoredact.core.model import AudioRedaction, BBox, Project, Span, VideoTrack
            p = Project(media=probe(media), author="selftest")
            p.add_audio_redaction(AudioRedaction(0.0, min(1.0, p.media.duration)))
            if p.media.has_video:
                p.add_video_track(VideoTrack(label="test", spans=[Span(0, 30, {0: BBox(.1, .1, .3, .3)})]))
            out = os.path.join(tempfile.gettempdir(), "videoredact_selftest.mp4" if p.media.has_video else "videoredact_selftest.m4a")
            r = export_project(p, ExportOptions(out, write_report=True, write_project=False))
            return f"{r['output']} ({r['elapsed_s']:.1f}s), report {os.path.basename(r.get('report_pdf', ''))}"
        step("export (first 1 s redacted)", _export)

        if "--detect" in sys.argv:
            def _detect():
                from videoredact.core.media import FrameReader
                from videoredact.vision.auto_detect import AutoDetectOptions, run_auto_detect
                with FrameReader(media) as r:
                    end = min(r.frame_count, 1800)
                    opts = AutoDetectOptions(labels=["face"], stride=5, end_frame=end, workers=2)
                    tracks = run_auto_detect(r, opts, progress=lambda p, m: None)
                return f"{len(tracks)} face track(s) in first {end} frames using 2 parallel workers"
            step("auto-detect (parallel workers)", _detect)

        if "--transcribe" in sys.argv:
            def _transcribe():
                from videoredact.audio.transcribe import Transcriber
                from videoredact.core.media import decode_audio
                from videoredact.paths import settings
                size = settings.get("whisper_model", "small")
                tr = Transcriber(size)
                audio, sr = decode_audio(media, sample_rate=16000, mono=True, start=0, duration=90)
                segs = tr.transcribe(audio[:, 0], duration=len(audio) / sr, language=None)
                words = sum(len(s.words) for s in segs)
                sample = " ".join(w.text for s in segs[:3] for w in s.words)[:120]
                return f"model {size}: {len(segs)} segments / {words} words in first 90 s; e.g. '{sample}'"
            step("speech recognition", _transcribe)

    lines.append("RESULT: " + ("ALL OK" if ok_all else "FAILURES (see log)"))
    text = "\n".join(lines)
    log.info("\n%s", text)
    try:
        sys.__stdout__.write(text + "\n")  # type: ignore[union-attr]
    except Exception:
        pass
    if getattr(sys, "frozen", False) and "--quiet" not in sys.argv:
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication(sys.argv)
            QMessageBox.information(None, "VideoRedact self-test", text)
        except Exception:
            pass
    return 0 if ok_all else 1


def main() -> None:
    # Must be first: parallel detection uses multiprocessing (spawn); in a frozen build the
    # child processes re-run this executable and freeze_support() hands control to them.
    import multiprocessing
    multiprocessing.freeze_support()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    _setup_logging()
    _install_excepthook()
    if "--selftest" in sys.argv:
        rest = [a for a in sys.argv[1:] if not a.startswith("--")]
        sys.exit(selftest(rest[0] if rest else None))
    import logging
    from videoredact import __version__
    logging.getLogger("app").info("VideoRedact %s starting (frozen=%s)", __version__, getattr(sys, "frozen", False))
    from PySide6.QtWidgets import QApplication
    from videoredact.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("VideoRedact")
    app.setOrganizationName("VideoRedact")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        win.open_path(sys.argv[1])
    win.schedule_update_check()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
