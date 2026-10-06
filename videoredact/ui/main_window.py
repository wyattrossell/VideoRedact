"""Main window: wires media playback, transcript, timeline, redaction panel
and the background tasks (transcribe, detect, track, export) together."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize, QTimer, QThread, Signal, QObject
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QSlider, QSplitter, QStatusBar, QStyle, QTabWidget,
                               QToolBar, QVBoxLayout, QWidget, QInputDialog, QDialog)

from videoredact import __version__, updater
from videoredact.audio.pii import suggest_pii
from videoredact.audio.transcribe import find_matches, words_in_range
from videoredact.core.media import MEDIA_FILTER, FrameReader, decode_audio, probe, sha256_file
from videoredact.core.model import (AudioRedaction, AudioStyle, BBox, Project, Span, VideoStyle, VideoTrack, Word)
from videoredact.paths import settings
from .dialogs import AutoDetectDialog, ExportDialog, NewBoxDialog, SettingsDialog
from .panels import AUDIO_STYLE_NAMES, VIDEO_STYLE_NAMES, PIIDialog, RedactionPanel
from .player import MediaPlayer
from .timeline import Timeline
from .transcript_view import TranscriptView
from .video_view import VideoView
from .workers import run_task
from .jobs import JobsPanel

PROJECT_FILTER = "VideoRedact project (*.vrproj)"


def fmt_time(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:06.3f}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"VideoRedact {__version__}")
        self.resize(1400, 860)
        self.project: Optional[Project] = None
        self.dirty = False
        self._audio_only_frame: Optional[np.ndarray] = None

        # ---- media player (own implementation, see ui/player.py) ----
        self.video = VideoView()
        self.player = MediaPlayer()
        self.player.frameReady.connect(self.video.show_still)
        self.player.positionChanged.connect(self._on_position)
        self.player.stateChanged.connect(self._on_state)
        self.player.error.connect(lambda m: self.statusBar().showMessage(m, 10000))

        # ---- widgets ----
        self.transcript = TranscriptView()
        self.panel = RedactionPanel()
        self.timeline = Timeline()
        tabs = QTabWidget()
        tabs.addTab(self.transcript, "Transcript")
        tabs.addTab(self.panel, "Redactions")
        self.tabs = tabs

        transport = self._build_transport()
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.video, 1)
        cl.addWidget(transport)
        cl.addWidget(self.timeline)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(center)
        split.addWidget(tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([900, 500])
        self.setCentralWidget(split)
        self.jobs = JobsPanel(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.jobs)
        self.setStatusBar(QStatusBar())
        self._build_menu()
        self._connect()
        self._update_enabled()

    # ------------------------------------------------------------------ UI
    def _build_transport(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(4, 2, 4, 2)
        st = self.style()
        self.btn_play = QPushButton(st.standardIcon(QStyle.SP_MediaPlay), "")
        self.btn_play.setToolTip("Play / pause (Space)")
        self.btn_play.clicked.connect(self.toggle_play)
        b_prev = QPushButton(st.standardIcon(QStyle.SP_MediaSeekBackward), "")
        b_prev.setToolTip("Previous frame (Left)")
        b_prev.clicked.connect(lambda: self.step_frames(-1))
        b_next = QPushButton(st.standardIcon(QStyle.SP_MediaSeekForward), "")
        b_next.setToolTip("Next frame (Right)")
        b_next.clicked.connect(lambda: self.step_frames(1))
        self.time_lbl = QLabel("0:00:00.000 / 0:00:00.000   frame 0")
        self.time_lbl.setMinimumWidth(300)
        self.preview_cb = QPushButton("Preview redactions")
        self.preview_cb.setCheckable(True)
        self.preview_cb.setChecked(True)
        self.preview_cb.toggled.connect(self._toggle_preview)
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(80)
        self.vol.setMaximumWidth(100)
        self.vol.valueChanged.connect(lambda v: self.player.set_volume(v / 100))
        for x in (b_prev, self.btn_play, b_next):
            h.addWidget(x)
        h.addWidget(self.time_lbl)
        h.addStretch(1)
        h.addWidget(self.preview_cb)
        h.addWidget(QLabel("Vol"))
        h.addWidget(self.vol)
        return w

    def _build_menu(self) -> None:
        mb = self.menuBar()
        st = self.style()

        def act(text, slot, shortcut=None, icon=None, checkable=False):
            a = QAction(text, self)
            if icon is not None:
                a.setIcon(st.standardIcon(icon))
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setCheckable(checkable)
            a.triggered.connect(slot)
            return a

        m = mb.addMenu("&File")
        self.a_open = act("&Open media…", self.open_media, "Ctrl+O", QStyle.SP_DialogOpenButton)
        self.a_open_proj = act("Open &project…", self.open_project, "Ctrl+Shift+O")
        self.a_save = act("&Save project", self.save_project, "Ctrl+S", QStyle.SP_DialogSaveButton)
        self.a_save_as = act("Save project &as…", self.save_project_as, "Ctrl+Shift+S")
        self.a_export = act("&Export redacted media…", self.export_media, "Ctrl+E", QStyle.SP_DialogApplyButton)
        for a in (self.a_open, self.a_open_proj, self.a_save, self.a_save_as):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.a_export)
        m.addSeparator()
        m.addAction(act("E&xit", self.close, "Ctrl+Q"))

        m = mb.addMenu("&Audio")
        self.a_transcribe = act("&Transcribe speech…", self.transcribe)
        self.a_find = act("&Find and redact word…", self.find_and_redact, "Ctrl+F")
        self.a_pii = act("Suggest &sensitive speech (PII)…", self.suggest_pii)
        self.a_range = act("Redact audio &range at playhead…", self.redact_audio_range, "Ctrl+R")
        for a in (self.a_transcribe, self.a_find, self.a_pii, self.a_range):
            m.addAction(a)

        m = mb.addMenu("&Video")
        self.a_detect = act("&Auto-detect faces / screens / documents…", self.auto_detect)
        self.a_track = act("&Track selected box", self.track_selected)
        self.a_clear_auto = act("Remove all a&utomatic detections", self.clear_auto)
        self.a_delete = act("&Delete selected redaction", self.delete_selected, "Delete")
        for a in (self.a_detect, self.a_track, self.a_clear_auto, self.a_delete):
            m.addAction(a)

        m = mb.addMenu("&Tools")
        m.addAction(act("&Settings…", self.open_settings, "Ctrl+,"))
        m.addAction(act("Download detection &models", self.download_models))
        m = mb.addMenu("&Help")
        m.addAction(act("&Quick guide", self.quick_guide, "F1"))
        m.addAction(act("Check for &updates…", lambda: self.check_for_updates(manual=True)))
        m.addAction(act("Open &log folder", self.open_log_folder))
        m.addAction(act("&About", self.about))

        tb = QToolBar("Main")
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        for a in (self.a_open, self.a_save, self.a_export):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.a_transcribe)
        tb.addAction(self.a_detect)
        tb.addSeparator()
        tb.addWidget(QLabel(" Audio style: "))
        self.c_astyle = QComboBox()
        for k, v in AUDIO_STYLE_NAMES.items():
            self.c_astyle.addItem(v, k)
        self.c_astyle.currentIndexChanged.connect(self._default_styles_changed)
        tb.addWidget(self.c_astyle)
        tb.addWidget(QLabel("  Video style: "))
        self.c_vstyle = QComboBox()
        for k, v in VIDEO_STYLE_NAMES.items():
            self.c_vstyle.addItem(v, k)
        self.c_vstyle.currentIndexChanged.connect(self._default_styles_changed)
        tb.addWidget(self.c_vstyle)
        tb.addSeparator()
        hint = QLabel("  Drag on the video to draw a box  ")
        hint.setStyleSheet("color: gray")
        tb.addWidget(hint)

    def _connect(self) -> None:
        self.video.boxDrawn.connect(self.on_box_drawn)
        self.video.trackSelected.connect(lambda tid: self._select("video", tid, from_video=True))
        self.video.boxEdited.connect(self.on_box_edited)
        self.video.deleteRequested.connect(self.delete_selected)
        self.video.frameShown.connect(self._on_frame_shown)
        self.transcript.seekRequested.connect(self.seek)
        self.transcript.redactWords.connect(self.redact_words)
        self.transcript.redactAllOf.connect(self.redact_all_of)
        self.transcript.unredactWords.connect(self.unredact_words)
        self.panel.seekRequested.connect(self.seek)
        self.panel.selectionChanged.connect(lambda k, i: self._select(k, i, from_panel=True))
        self.panel.changed.connect(self._project_changed)
        self.panel.deleteRequested.connect(self._delete)
        self.panel.trackRequested.connect(self.track_id)
        self.timeline.canvas.seekRequested.connect(self.seek)
        self.timeline.canvas.audioSelected.connect(lambda i: self._select("audio", i))
        self.timeline.canvas.trackSelected.connect(lambda i: self._select("video", i))
        self.timeline.canvas.rangeSelected.connect(self.redact_time_range)

    def _update_enabled(self) -> None:
        has = self.project is not None
        vid = has and self.project.media.has_video
        aud = has and self.project.media.has_audio
        for a in (self.a_save, self.a_save_as, self.a_export, self.a_range):
            a.setEnabled(has)
        for a in (self.a_transcribe,):
            a.setEnabled(aud)
        for a in (self.a_find, self.a_pii):
            a.setEnabled(has and bool(self.project.transcript))
        for a in (self.a_detect, self.a_track, self.a_clear_auto):
            a.setEnabled(vid)
        self.video.draw_mode = vid

    # ---------------------------------------------------------- project IO
    def open_path(self, path: str) -> None:
        if path.lower().endswith(".vrproj"):
            self._load_project(path)
        else:
            self._load_media(path)

    def open_media(self) -> None:
        if not self._confirm_discard():
            return
        p, _ = QFileDialog.getOpenFileName(self, "Open video or audio", settings.get("last_dir", ""), MEDIA_FILTER)
        if p:
            self._load_media(p)

    def _load_media(self, path: str) -> None:
        settings["last_dir"] = str(Path(path).parent)
        settings.save()
        try:
            info = probe(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot open", f"{path}\n\n{e}")
            return
        if not info.has_video and not info.has_audio:
            QMessageBox.critical(self, "Cannot open", "No audio or video stream found in this file.")
            return
        proj = Project(media=info, author=settings["author"],
                       default_audio_style=AudioStyle(settings["default_audio_style"]),
                       default_video_style=VideoStyle(settings["default_video_style"]),
                       beep_frequency=settings["beep_frequency"])
        proj.log("media_opened", f"{path} ({info.width}x{info.height} @ {info.fps:.3f} fps, {info.duration:.1f}s)")
        self._set_project(proj)
        # Hash the source in the background (needed for the report)
        self.jobs.run("Hashing source file",
                      lambda prog, cancel, partial: sha256_file(path, lambda f: prog(f, "SHA-256 of source")),
                      lambda h: self._set_hash(proj, h))

    def _set_hash(self, proj: Project, h: str) -> None:
        if self.project is proj:
            proj.media.sha256 = h
            self.statusBar().showMessage(f"Source SHA-256: {h}", 8000)

    def open_project(self) -> None:
        if not self._confirm_discard():
            return
        p, _ = QFileDialog.getOpenFileName(self, "Open project", settings.get("last_dir", ""), PROJECT_FILTER)
        if p:
            self._load_project(p)

    def _load_project(self, path: str) -> None:
        try:
            proj = Project.load(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot open project", str(e))
            return
        if not Path(proj.media.path).exists():
            QMessageBox.warning(self, "Media missing", f"The media file referenced by this project was not found:\n{proj.media.path}\n\nPlease locate it.")
            p, _ = QFileDialog.getOpenFileName(self, "Locate media", str(Path(path).parent), MEDIA_FILTER)
            if not p:
                return
            proj.media.path = p
            proj.log("media_relinked", p)
        proj.author = proj.author or settings["author"]
        proj.log("project_opened", path)
        self._set_project(proj)

    def _set_project(self, proj: Project) -> None:
        self.player.stop()
        self.project = proj
        self.dirty = False
        self.video.selected_track_id = None
        self.c_astyle.blockSignals(True)
        self.c_vstyle.blockSignals(True)
        self.c_astyle.setCurrentIndex(list(AUDIO_STYLE_NAMES).index(proj.default_audio_style))
        self.c_vstyle.setCurrentIndex(list(VIDEO_STYLE_NAMES).index(proj.default_video_style))
        self.c_astyle.blockSignals(False)
        self.c_vstyle.blockSignals(False)
        m = proj.media
        self.video.set_media_props(m.width or 640, m.height or 360, m.fps or 30.0)
        if m.has_video:
            self._audio_only_frame = None
        else:
            # audio-only: show a waveform-ish placeholder so the view is not empty
            self._audio_only_frame = self._waveform_image(m.path, 960, 300)
            self.video.set_media_props(960, 300, 30.0)
            self.video.show_still(self._audio_only_frame, 0)
        self.player.load(m, proj)
        self.transcript.set_project(proj)
        self.panel.set_project(proj)
        self.timeline.set_project(proj)
        self.video.set_tracks(proj.video_tracks, proj.default_video_style)
        self.setWindowTitle(f"VideoRedact {__version__} — {Path(m.path).name}")
        self._update_enabled()
        self.statusBar().showMessage(
            f"Loaded {Path(m.path).name}: {fmt_time(m.duration)}, "
            + (f"{m.width}x{m.height} @ {m.fps:.2f} fps, " if m.has_video else "audio only, ")
            + (f"{m.sample_rate} Hz {m.channels} ch" if m.has_audio else "no audio"), 10000)
        self.seek(0.0)

    def _waveform_image(self, path: str, w: int, h: int) -> np.ndarray:
        img = np.full((h, w, 3), 30, dtype=np.uint8)
        try:
            samples, sr = decode_audio(path, sample_rate=8000, mono=True)
            x = samples[:, 0]
            n = len(x)
            if n:
                per = max(1, n // w)
                mx = np.array([np.abs(x[i * per:(i + 1) * per]).max() if (i + 1) * per <= n else 0 for i in range(w)])
                for i, v in enumerate(mx):
                    a = int(v * (h / 2 - 4))
                    cv2.line(img, (i, h // 2 - a), (i, h // 2 + a), (200, 170, 90), 1)
        except Exception:
            pass
        cv2.putText(img, "Audio file", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        return img

    def save_project(self) -> None:
        if not self.project:
            return
        if not self.project.path:
            return self.save_project_as()
        self._save_to(self.project.path)

    def save_project_as(self) -> None:
        if not self.project:
            return
        default = self.project.path or str(Path(self.project.media.path).with_suffix(".vrproj"))
        p, _ = QFileDialog.getSaveFileName(self, "Save project", default, PROJECT_FILTER)
        if p:
            self._save_to(p)

    def _save_to(self, path: str) -> None:
        try:
            self.project.log("project_saved", path)
            self.project.save(path)
            self.dirty = False
            self.statusBar().showMessage(f"Saved {path}", 5000)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(e))

    def _confirm_discard(self) -> bool:
        if not self.project or not self.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save the current project first?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            self.save_project()
            return not self.dirty
        return r == QMessageBox.Discard

    def closeEvent(self, event):
        if self.jobs.running and QMessageBox.question(
                self, "Jobs running", f"{self.jobs.running} background job(s) are still running. Cancel them and quit?") != QMessageBox.Yes:
            event.ignore()
            return
        if self._confirm_discard():
            self.jobs.cancel_all()
            self.jobs.wait_all(3000)
            self.player.stop()
            event.accept()
        else:
            event.ignore()

    def _project_changed(self) -> None:
        self.dirty = True
        self.panel.rebuild(self.panel.current()[1] or None)
        self.timeline.refresh()
        self.transcript.refresh_marks()
        self.video.set_tracks(self.project.video_tracks, self.project.default_video_style)

    def _default_styles_changed(self) -> None:
        if not self.project:
            return
        self.project.default_audio_style = self.c_astyle.currentData()
        self.project.default_video_style = self.c_vstyle.currentData()
        self.project.log("default_styles", f"audio={self.project.default_audio_style.value} video={self.project.default_video_style.value}")
        self._project_changed()

    # ------------------------------------------------------------ playback
    @property
    def fps(self) -> float:
        return (self.project.media.fps if self.project and self.project.media.fps else 30.0)

    def current_time(self) -> float:
        return self.player.position_s()

    def current_frame(self) -> int:
        return self.player.current_frame()

    def toggle_play(self) -> None:
        if self.project:
            self.player.toggle()

    def step_frames(self, n: int) -> None:
        if self.project:
            self.player.step(n)

    def seek(self, t: float) -> None:
        if self.project:
            self.player.seek(t)

    def _on_state(self, playing: bool) -> None:
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay))

    def _on_position(self, ms: int) -> None:
        t = ms / 1000.0
        self.timeline.set_playhead(t)
        self.transcript.set_time(t)
        f = int(t * self.fps + 1e-6)
        dur = self.project.media.duration if self.project else 0.0
        self.time_lbl.setText(f"{fmt_time(t)} / {fmt_time(dur)}   frame {f}")
        if self.project and not self.project.media.has_video:
            self.video.set_frame_index(f)

    def _on_frame_shown(self, idx: int) -> None:
        pass

    def _toggle_preview(self, on: bool) -> None:
        self.video.preview_redactions = on
        self.player.preview_redactions = on
        self.video._render()

    # ------------------------------------------------------------ selection
    def _select(self, kind: str, rid: str, from_video: bool = False, from_panel: bool = False) -> None:
        if kind == "video":
            self.video.select_track(rid)
            if not from_panel:
                self.panel.select("video", rid)
            self.timeline.canvas.selected_id = rid
        elif kind == "audio":
            if not from_panel:
                self.panel.select("audio", rid)
            self.video.select_track(None)
            self.timeline.canvas.selected_id = rid
        else:
            self.video.select_track(None)
            self.timeline.canvas.selected_id = None
        self.timeline.canvas.update()

    def delete_selected(self) -> None:
        kind, rid = self.panel.current()
        if not kind and self.video.selected_track_id:
            kind, rid = "video", self.video.selected_track_id
        if kind:
            self._delete(kind, rid)

    def _delete(self, kind: str, rid: str) -> None:
        if not self.project:
            return
        if kind == "audio":
            self.project.remove_audio_redaction(rid)
        else:
            self.project.remove_video_track(rid)
            self.video.selected_track_id = None
        self._project_changed()

    # ----------------------------------------------------------- audio ops
    def redact_words(self, words: list[Word], source: str = "manual", reason: str = "") -> None:
        if not words or not self.project:
            return
        text = " ".join(w.text for w in words)
        r = AudioRedaction(words[0].start, words[-1].end, text=text, reason=reason, source=source)
        self.project.add_audio_redaction(r)
        self._project_changed()
        self.statusBar().showMessage(f"Redacted “{text}” ({fmt_time(r.start)} – {fmt_time(r.end)})", 5000)

    def redact_all_of(self, phrase: str) -> None:
        if not phrase or not self.project:
            return
        hits = find_matches(self.project.transcript, phrase)
        n = 0
        for words in hits:
            if all(self.project.word_is_redacted(w) for w in words):
                continue
            text = " ".join(w.text for w in words)
            self.project.add_audio_redaction(AudioRedaction(words[0].start, words[-1].end, text=text,
                                                            reason=f"all occurrences of '{phrase}'", source="word_match"))
            n += 1
        self._project_changed()
        self.statusBar().showMessage(f"Redacted {n} occurrence(s) of “{phrase}” ({len(hits)} found)", 6000)

    def unredact_words(self, words: list[Word]) -> None:
        if not self.project:
            return
        t0, t1 = words[0].start, words[-1].end
        for r in [r for r in self.project.audio_redactions if r.end > t0 and r.start < t1]:
            self.project.remove_audio_redaction(r.id)
        self._project_changed()

    def find_and_redact(self) -> None:
        self.tabs.setCurrentWidget(self.transcript)
        self.transcript.search.setFocus()
        self.transcript.search.selectAll()

    def redact_audio_range(self) -> None:
        if not self.project:
            return
        t = self.current_time()
        dur, ok = QInputDialog.getDouble(self, "Redact audio range", f"Redact audio starting at {fmt_time(t)} for how many seconds?",
                                         2.0, 0.05, 3600.0, 2)
        if not ok:
            return
        words = words_in_range(self.project.transcript, t, t + dur)
        r = AudioRedaction(t, t + dur, text=" ".join(w.text for w in words), source="range")
        self.project.add_audio_redaction(r)
        self._project_changed()

    def redact_time_range(self, t0: float, t1: float) -> None:
        """Audio redaction from a timeline drag (works with or without a transcript)."""
        if not self.project:
            return
        words = words_in_range(self.project.transcript, t0, t1)
        r = AudioRedaction(t0, t1, text=" ".join(w.text for w in words), source="range")
        self.project.add_audio_redaction(r)
        self._project_changed()
        self._select("audio", r.id)
        self.statusBar().showMessage(f"Audio redacted {fmt_time(t0)} – {fmt_time(t1)}", 5000)

    def transcribe(self) -> None:
        if not self.project or not self.project.media.has_audio:
            return
        if self.project.transcript and QMessageBox.question(
                self, "Transcribe", "Replace the existing transcript? Existing audio redactions are kept (they are time-based).") != QMessageBox.Yes:
            return
        path = self.project.media.path
        size = settings["whisper_model"]
        lang = settings["language"] or None
        threads = settings["cpu_threads"]
        vad = bool(settings.get("vad", False))
        duration = self.project.media.duration
        proj = self.project
        proj.transcript = []
        self.transcript.rebuild()
        self.tabs.setCurrentWidget(self.transcript)

        def job(prog, cancel, partial):
            prog(0.0, f"Loading speech model '{size}'…")
            from videoredact.audio.transcribe import Transcriber
            tr = Transcriber(size, cpu_threads=threads)
            prog(0.02, "Decoding audio…")
            audio, sr = decode_audio(path, sample_rate=16000, mono=True)
            if cancel():
                return None
            prog(0.03, "Listening… first words appear as soon as speech is found")
            return tr.transcribe(audio[:, 0], duration=duration, language=lang, progress=prog, cancel=cancel,
                                 vad=vad, on_segment=partial)

        def on_partial(seg):
            if self.project is proj:
                proj.transcript.append(seg)
                self.transcript.append_segment(seg)
                self._update_enabled()

        def done(segments):
            if segments is None or self.project is not proj:
                return
            proj.transcript = segments
            proj.transcript_model = f"faster-whisper {size} (int8, CPU{', VAD' if vad else ''})"
            proj.log("transcribed", f"{len(segments)} segments, model {size}")
            self.transcript.rebuild()
            self._update_enabled()
            self.dirty = True
            n_words = sum(len(s.words) for s in segments)
            msg = f"Transcription complete: {len(segments)} segments, {n_words} words"
            if not segments:
                msg += " - no speech recognised. Check the audio track plays, or try a larger model in Settings."
            self.statusBar().showMessage(msg, 12000)

        self.jobs.run("Transcribing speech", job, done, on_partial=on_partial,
                      on_cancel=lambda: self.statusBar().showMessage("Transcription cancelled (partial transcript kept)", 6000))

    def suggest_pii(self) -> None:
        if not self.project or not self.project.transcript:
            return
        sugg = [s for s in suggest_pii(self.project.transcript)
                if not all(self.project.word_is_redacted(w) for w in s.words)]
        if not sugg:
            QMessageBox.information(self, "Suggestions", "No new sensitive speech patterns found.\n"
                                    "Review the transcript manually; the rules are simple heuristics.")
            return
        dlg = PIIDialog(sugg, self)
        if dlg.exec() == QDialog.Accepted:
            for s in dlg.chosen():
                self.project.add_audio_redaction(AudioRedaction(s.start, s.end, text=s.text,
                                                                reason=f"PII: {s.kind}", source="pii"))
            self._project_changed()

    # ----------------------------------------------------------- video ops
    def on_box_drawn(self, bbox: BBox) -> None:
        if not self.project or not self.project.media.has_video:
            return
        self.player.pause()
        frame = self.current_frame()
        dlg = NewBoxDialog(settings, self.current_time(), self.project.media.duration, self)
        if dlg.exec() != QDialog.Accepted:
            return
        label = dlg.label.currentText().strip() or "other"
        track = VideoTrack(label=label, style=dlg.style.currentData(), shape=dlg.shape.currentData(),
                           pad=settings["video_pad"], source="manual")
        mode = dlg.mode()
        if mode == "keyframe":
            track.add_keyframe(frame, bbox)
            self.project.add_video_track(track)
            self._project_changed()
            self._select("video", track.id)
        elif mode == "static":
            f0 = int(round(dlg.t0.value() * self.fps))
            f1 = int(round(dlg.t1.value() * self.fps))
            if f1 < f0:
                f0, f1 = f1, f0
            track.spans = [Span(f0, f1, {f0: bbox})]
            track.note = "fixed box"
            self.project.add_video_track(track)
            self._project_changed()
            self._select("video", track.id)
        else:
            end_frame = min(self.project.media.frame_count - 1, int(round(dlg.track_end_time * self.fps))) if self.project.media.frame_count else None
            self._run_tracking(frame, bbox, label, track, step=dlg.track_step, backward=dlg.track_backward, end_frame=end_frame)

    def _run_tracking(self, frame: int, bbox: BBox, label: str, proto: VideoTrack, replace_id: Optional[str] = None,
                      step: Optional[int] = None, backward: Optional[bool] = None, end_frame: Optional[int] = None) -> None:
        """Track in the background. The track is added immediately with the drawn box and
        fills in live as the tracker advances, so the UI stays usable."""
        proj = self.project
        path = proj.media.path
        det_label = label if label in ("face", "screen", "document", "phone", "person") else None
        face_conf, obj_conf, model, threads = settings["face_conf"], settings["detect_conf"], settings.get("detector_model", "yolox_tiny"), settings["cpu_threads"]
        face_model = settings.get("face_detector", "yunet")
        from videoredact.vision.tracker import TrackOptions
        opts = TrackOptions(step=int(step or settings.get("track_step", 2)),
                            backward=bool(settings.get("track_backward", True) if backward is None else backward),
                            end_frame=end_frame)
        # live placeholder track
        live = VideoTrack(label=label, style=proto.style, shape=proto.shape, pad=proto.pad, source="tracked",
                          note="tracking in progress…")
        live.add_keyframe(frame, bbox)
        if replace_id:
            proj.remove_video_track(replace_id)
        proj.add_video_track(live)
        self._project_changed()
        self._select("video", live.id)

        def job(prog, cancel, partial):
            from videoredact.vision.tracker import ObjectTracker
            det = None
            if det_label:
                try:
                    from videoredact.vision.detector import CombinedDetector
                    prog(0.0, "Loading detector…")
                    det = CombinedDetector([det_label], face_conf, obj_conf, model, threads, face_model=face_model)
                except Exception:
                    det = None
            with FrameReader(path) as reader:
                return ObjectTracker(reader, det, opts).track(frame, bbox, label=det_label or label,
                                                              progress=prog, cancel=cancel, on_update=partial)

        def on_partial(spans):
            if self.project is proj and proj.get_track(live.id) is live:
                live.spans = spans
                live.merge_spans()
                self.timeline.refresh()
                self.video.set_tracks(proj.video_tracks, proj.default_video_style)

        def done(track: VideoTrack):
            if track is None or self.project is not proj or proj.get_track(live.id) is not live:
                return
            live.spans = track.spans
            live.note = f"tracked from frame {frame}; {len(track.spans)} visible span(s); every {opts.step} frame(s)"
            proj.log("tracked", f"{label} {live.total_frames()} frames in {len(live.spans)} span(s)")
            self._project_changed()
            self.statusBar().showMessage(f"Tracked '{label}': {live.total_frames()} frames in {len(live.spans)} span(s)", 8000)

        def on_cancel():
            if self.project is proj:
                live.note = "tracking cancelled (partial result kept)"
                self._project_changed()

        self.jobs.run(f"Tracking '{label}'", job, done, on_partial=on_partial, on_cancel=on_cancel)

    def on_box_edited(self, tid: str, bbox: BBox) -> None:
        t = self.project.get_track(tid) if self.project else None
        if not t:
            return
        f = self.current_frame()
        t.set_keyframe_extend(f, bbox)
        if t.source == "auto":
            t.source = "auto+edited"
        self.project.log("keyframe_edited", f"{t.label} frame {f}")
        self._project_changed()

    def track_selected(self) -> None:
        kind, rid = self.panel.current()
        tid = rid if kind == "video" else self.video.selected_track_id
        if tid:
            self.track_id(tid)

    def track_id(self, tid: str) -> None:
        t = self.project.get_track(tid) if self.project else None
        if not t:
            return
        f = self.current_frame()
        bb = t.bbox_at(f)
        if bb is None:
            f = t.first_frame()
            bb = t.bbox_at(f)
        if bb is None:
            return
        if QMessageBox.question(self, "Track object",
                                f"Track '{t.label}' starting from its box at frame {f}?\n"
                                "This replaces the existing track with the tracked result.") != QMessageBox.Yes:
            return
        self._run_tracking(f, bb, t.label, t, replace_id=tid)

    def auto_detect(self) -> None:
        if not self.project or not self.project.media.has_video:
            return
        dlg = AutoDetectDialog(settings, False, self)
        dlg.set_estimate(self.project.media.frame_count, self.fps)
        dlg.stride.valueChanged.connect(lambda _: dlg.set_estimate(self.project.media.frame_count, self.fps))
        if dlg.exec() != QDialog.Accepted:
            return
        labels = dlg.labels()
        if not labels:
            return
        settings["detect_classes"] = labels
        settings["detect_stride"] = dlg.stride.value()
        settings.save()
        path = self.project.media.path
        from videoredact.vision.auto_detect import AutoDetectOptions, run_auto_detect
        opts = AutoDetectOptions(labels=labels, stride=dlg.stride.value(), face_conf=settings["face_conf"],
                                 obj_conf=settings["detect_conf"], obj_model=settings.get("detector_model", "yolox_tiny"),
                                 threads=settings["cpu_threads"])
        replace = dlg.replace.isChecked()

        opts.face_model = settings.get("face_detector", "yunet")
        proj = self.project

        def job(prog, cancel, partial):
            prog(0.0, "Loading detection models…")
            with FrameReader(path) as reader:
                return run_auto_detect(reader, opts, prog, cancel)

        def done(tracks):
            if tracks is None or self.project is not proj:
                return
            if replace:
                for t in [t for t in self.project.video_tracks if t.source == "auto"]:
                    self.project.video_tracks.remove(t)
            for t in tracks:
                t.pad = settings["video_pad"]
                self.project.add_video_track(t)
            self.project.log("auto_detect", f"{len(tracks)} regions, labels={labels}, stride={opts.stride}")
            self._project_changed()
            self.tabs.setCurrentWidget(self.panel)
            QMessageBox.information(self, "Detection complete",
                                    f"Found {len(tracks)} region(s): " +
                                    ", ".join(f"{labels.count(l) if False else sum(1 for t in tracks if t.label == l)} {l}" for l in labels) +
                                    ".\n\nPlay the video with 'Preview redactions' on to review. "
                                    "Delete false hits in the Redactions panel; draw boxes for anything missed.")

        self.jobs.run("Auto-detecting objects", job, done)

    def clear_auto(self) -> None:
        if not self.project:
            return
        n = sum(1 for t in self.project.video_tracks if t.source == "auto")
        if n and QMessageBox.question(self, "Remove", f"Remove {n} automatic detection(s)?") == QMessageBox.Yes:
            self.project.video_tracks = [t for t in self.project.video_tracks if t.source != "auto"]
            self.project.log("auto_cleared", str(n))
            self._project_changed()

    # --------------------------------------------------------------- export
    def export_media(self) -> None:
        if not self.project:
            return
        self.player.pause()
        dlg = ExportDialog(self.project, settings, self)
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.options()
        opts.audio_pad_s = settings["audio_pad_s"]
        if not opts.output_path:
            return
        if os.path.abspath(opts.output_path) == os.path.abspath(self.project.media.path):
            QMessageBox.critical(self, "Export", "The output must be a different file from the source.")
            return
        settings["export_crf"], settings["export_preset"] = opts.crf, opts.preset
        settings.save()
        proj = self.project

        def job(prog, cancel, partial):
            from videoredact.core.export import ExportCancelled, export_project
            try:
                return export_project(proj, opts, prog, cancel)
            except ExportCancelled:
                return None

        def on_cancel():
            self.statusBar().showMessage("Export cancelled", 5000)
            try:
                os.remove(opts.output_path)
            except OSError:
                pass

        def done(res):
            if res is None:
                on_cancel()
                return
            self.dirty = True
            lines = [f"Redacted media written to:\n{res['output']}", f"SHA-256: {res['output_sha256']}"]
            if "report_pdf" in res:
                lines.append(f"\nReport: {res['report_pdf']}\n        {res['report_csv']}")
            if "project" in res:
                lines.append(f"Project: {res['project']}")
            lines.append(f"\nTook {res['elapsed_s']:.0f} s.")
            box = QMessageBox(QMessageBox.Information, "Export complete", "\n".join(lines), parent=self)
            open_btn = box.addButton("Open folder", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Ok)
            box.exec()
            if box.clickedButton() == open_btn:
                os.startfile(str(Path(res["output"]).parent))  # type: ignore[attr-defined]

        self.jobs.run("Exporting redacted media", job, done, on_cancel=on_cancel)

    # ---------------------------------------------------------------- misc
    def open_settings(self) -> None:
        dlg = SettingsDialog(settings, self.project, self)
        if dlg.exec() == QDialog.Accepted:
            dlg.apply()
            if self.project:
                self.c_astyle.setCurrentIndex(list(AUDIO_STYLE_NAMES).index(self.project.default_audio_style))
                self.c_vstyle.setCurrentIndex(list(VIDEO_STYLE_NAMES).index(self.project.default_video_style))
                self._project_changed()

    def download_models(self) -> None:
        from videoredact.vision import models as reg

        def job(prog, cancel):
            names = list(reg.MODELS)
            for i, n in enumerate(names):
                if cancel():
                    return None
                prog(i / len(names), f"Checking {n}…")
                reg.ensure_model(n, lambda p, m: prog((i + p) / len(names), m))
            return True

        run_task(self, "Downloading detection models…", job,
                 lambda ok: ok and QMessageBox.information(self, "Models", f"All detection models are available in\n{reg.models_dir()}"))

    # ------------------------------------------------------------- updates
    def schedule_update_check(self) -> None:
        """Called once after the window is shown. Checks GitHub Releases in the
        background a few seconds after startup, if enabled in settings."""
        if settings.get("check_updates", True):
            QTimer.singleShot(4000, lambda: self.check_for_updates(manual=False))

    def check_for_updates(self, manual: bool) -> None:
        class _Checker(QThread):
            result = Signal(object, str)

            def run(self_):
                try:
                    self_.result.emit(updater.fetch_latest(), "")
                except Exception as e:  # noqa: BLE001
                    self_.result.emit(None, str(e))

        self._update_checker = _Checker(self)
        self._update_checker.result.connect(lambda info, err: self._on_update_result(info, err, manual))
        self._update_checker.start()
        if manual:
            self.statusBar().showMessage("Checking for updates…", 5000)

    def _on_update_result(self, info, err: str, manual: bool) -> None:
        if err:
            if manual:
                QMessageBox.warning(self, "Check for updates",
                                    f"Could not reach GitHub to check for updates.\n\n{err}\n\n"
                                    f"Releases page: {updater.RELEASES_PAGE}")
            return
        if info is None or not updater.is_newer(info.version):
            if manual:
                QMessageBox.information(self, "Check for updates",
                                        f"You are running the latest version ({__version__}).")
            return
        if not manual and settings.get("skip_version") == info.version:
            return
        notes = (info.notes or "").strip()
        if len(notes) > 1200:
            notes = notes[:1200] + "…"
        box = QMessageBox(QMessageBox.Information, "Update available",
                          f"VideoRedact {info.version} is available (you have {__version__}).\n\n"
                          + (f"{notes}\n\n" if notes else "")
                          + "Install it now? The application will close, update and reopen.", parent=self)
        b_install = box.addButton("Install now", QMessageBox.AcceptRole)
        b_later = box.addButton("Later", QMessageBox.RejectRole)
        b_skip = box.addButton("Skip this version", QMessageBox.DestructiveRole)
        box.setDefaultButton(b_install)
        box.exec()
        if box.clickedButton() == b_skip:
            settings["skip_version"] = info.version
            settings.save()
        elif box.clickedButton() == b_install:
            self._install_update(info)

    def _install_update(self, info) -> None:
        if not info.url:
            QMessageBox.information(self, "Update", f"This release has no installer attached yet.\nSee {info.page}")
            return
        if not self._confirm_discard():
            return

        def job(prog, cancel):
            return updater.download(info, prog, cancel)

        def done(path):
            if not path:
                return
            self.dirty = False
            try:
                updater.launch_installer(path, restart=True)
            except Exception as e:  # noqa: BLE001
                QMessageBox.critical(self, "Update", f"Could not start the installer:\n{e}\n\nIt was saved to:\n{path}")
                return
            QApplication.instance().quit()

        run_task(self, f"Downloading VideoRedact {info.version}…", job, done)

    def open_log_folder(self) -> None:
        from videoredact.paths import log_dir
        os.startfile(str(log_dir()))  # type: ignore[attr-defined]

    def quick_guide(self) -> None:
        QMessageBox.information(self, "Quick guide", """
<b>1. Open</b> a video or audio file (File ▸ Open media).<br>
<b>2. Audio:</b> Audio ▸ Transcribe. In the Transcript tab select words, right-click ▸ Redact,
or ▸ Redact every occurrence. Use the search box to find a word everywhere.
Audio ▸ Suggest sensitive speech finds phone numbers, names after cues, addresses, dates of birth.<br>
<b>3. Video:</b> Video ▸ Auto-detect finds faces, screens and documents. Drag a box on the video around
anything else (a license plate, a tattoo, a notepad) and choose <i>Track this object</i>: it is followed
forward and backward and re-acquired when it leaves and re-enters the frame.<br>
<b>4. Review:</b> Play with <i>Preview redactions</i> on. Redacted audio is muted during preview.
Drag a box to fix its position: that adds a keyframe. Delete false hits in the Redactions tab.<br>
<b>5. Export</b> (Ctrl+E): writes the redacted file plus a PDF/CSV report with SHA-256 hashes and the project file.<br><br>
<b>Keys:</b> Space play/pause · ←/→ frame step · Delete remove selected · Ctrl+wheel zoom timeline · double-click video to fit.
""")

    def about(self) -> None:
        QMessageBox.about(self, "About VideoRedact",
                          f"<b>VideoRedact {__version__}</b><br>Free, offline video and audio redaction for law enforcement.<br>"
                          "MIT License. Runs entirely on this computer; no media data leaves the machine "
                          "(the only network access is the optional update check against GitHub).<br><br>"
                          "Components: FFmpeg (LGPL build), OpenCV, ONNX Runtime, faster-whisper, YuNet, YOLOX, Qt/PySide6 (LGPL).")

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Space and not isinstance(QApplication.focusWidget(), (type(self.transcript.search),)):
            self.toggle_play()
        elif k == Qt.Key_Left:
            self.step_frames(-(int(self.fps) if event.modifiers() & Qt.ShiftModifier else 1))
        elif k == Qt.Key_Right:
            self.step_frames(int(self.fps) if event.modifiers() & Qt.ShiftModifier else 1)
        else:
            super().keyPressEvent(event)
