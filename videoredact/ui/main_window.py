"""Main window: wires media playback, transcript, timeline, redaction panel
and the background tasks (transcribe, detect, track, export) together."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, QEvent, QSize, QTimer, QThread, Signal, QObject, QUrl
from PySide6.QtGui import QAction, QKeySequence, QIcon, QDesktopServices
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QSlider, QSplitter, QStatusBar, QStyle, QTabWidget,
                               QToolBar, QVBoxLayout, QWidget, QInputDialog, QDialog)

from videoredact import __version__, updater
from videoredact.audio.pii import suggest_pii
from videoredact.audio.transcribe import find_matches, words_in_range
from videoredact.core.media import MEDIA_FILTER, FrameReader, decode_audio, probe, sha256_file
from videoredact.core.model import (AudioRedaction, AudioStyle, BBox, Project, Shape, Span, VideoStyle, VideoTrack,
                                    Word, as_enum)
from videoredact.paths import settings
from .dialogs import AutoDetectDialog, BatchDialog, ExportDialog, NewBoxDialog, SettingsDialog
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
        self.a_batch = act("&Batch process files…", self.batch_process, "Ctrl+B")
        for a in (self.a_open, self.a_open_proj, self.a_save, self.a_save_as):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.a_export)
        m.addAction(self.a_batch)
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
        self.a_track = act("&Track selected box (forward and backward)", self.track_selected, "T")
        self.a_join = act("&Join the next drawn box to the selected track", self.join_next_box, "J")
        self.a_quick = act("Track every new box immediately (Shift+drag asks first)", self.toggle_quick_box, checkable=True)
        self.a_quick.setChecked(bool(settings.get("quick_box", True)))
        self.a_clear_auto = act("Remove all a&utomatic detections", self.clear_auto)
        self.a_delete = act("&Delete selected redaction", self.delete_selected, "Delete")
        self.a_undo = act("&Undo delete", self.undo_delete, "Ctrl+Z")
        self.a_trim_start = act("Trim track start to playhead", lambda: self.trim_selected("start"), "[")
        self.a_trim_end = act("Trim track end to playhead", lambda: self.trim_selected("end"), "]")
        self.a_extend = act("Extend track to playhead", self.extend_selected, "=")
        self.a_split = act("Split track at playhead", self.split_selected, "X")
        self.a_toggle = act("Enable / disable selected track", self.toggle_selected, "E")
        self.a_next = act("Next redaction", lambda: self.panel.step(1), "N")
        self.a_prev = act("Previous redaction", lambda: self.panel.step(-1), "Shift+N")
        self.a_next_flag = act("Next flagged track", lambda: self.panel.next_flagged(1), "F")
        self.a_prev_flag = act("Previous flagged track", lambda: self.panel.next_flagged(-1), "Shift+F")
        for a in (self.a_detect, self.a_track, self.a_join, self.a_quick, self.a_clear_auto, self.a_delete, self.a_undo):
            m.addAction(a)
        m.addSeparator()
        for a in (self.a_trim_start, self.a_trim_end, self.a_extend, self.a_split, self.a_toggle,
                  self.a_next, self.a_prev, self.a_next_flag, self.a_prev_flag):
            m.addAction(a)

        m = mb.addMenu("&Tools")
        m.addAction(act("&Settings…", self.open_settings, "Ctrl+,"))
        m.addAction(act("Check detection &models…", self.download_models))
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
        QApplication.instance().installEventFilter(self)   # transport keys from any panel (see eventFilter)

    def _update_enabled(self) -> None:
        has = self.project is not None
        vid = has and self.project.media.has_video
        aud = has and self.project.media.has_audio
        for a in (self.a_save, self.a_save_as, self.a_export):
            a.setEnabled(has)
        for a in (self.a_transcribe, self.a_range):
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
        self.c_astyle.setCurrentIndex(list(AUDIO_STYLE_NAMES).index(as_enum(AudioStyle, proj.default_audio_style)))
        self.c_vstyle.setCurrentIndex(list(VIDEO_STYLE_NAMES).index(as_enum(VideoStyle, proj.default_video_style)))
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
        self.project.default_audio_style = as_enum(AudioStyle, self.c_astyle.currentData())
        self.project.default_video_style = as_enum(VideoStyle, self.c_vstyle.currentData())
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

    def _delete(self, kind: str, rid: str, confirm: bool = True) -> None:
        if not self.project:
            return
        if kind == "audio":
            self.project.remove_audio_redaction(rid)
        else:
            t = next((x for x in self.project.video_tracks if x.id == rid), None)
            if t is not None:
                # deletes are undoable (Ctrl+Z) so reviewing is one key per false hit, no dialog
                self._push_undo(t)
            self.project.remove_video_track(rid)
            self.video.selected_track_id = None
            if t is not None:
                self.statusBar().showMessage(f"Deleted '{t.label}' track ({t.total_frames()} frames). Ctrl+Z undoes.", 6000)
        self._project_changed()

    def _push_undo(self, t: VideoTrack, kind: str = "deleted") -> None:
        """kind: "deleted" (undo re-adds the track) or "replaced" (t is a snapshot; undo puts it back)."""
        st = self.__dict__.setdefault("_undo_stack", [])
        st.append((kind, t))
        del st[:-20]

    def undo_delete(self) -> None:
        st = self.__dict__.get("_undo_stack") or []
        if not st or not self.project:
            return
        kind, t = st.pop()
        cur = self.project.get_track(t.id)
        if kind == "replaced":
            if cur is None:
                return
            cur.spans, cur.note = t.spans, t.note
            self.project.log("video_track_restored", f"{t.label} join undone")
            self._project_changed()
            self._select("video", t.id)
            self.statusBar().showMessage(f"Undid the join on the '{t.label}' track.", 4000)
            return
        if cur is not None:
            return
        self.project.add_video_track(t)
        self.project.log("video_track_restored", f"{t.label} frames {t.first_frame()}-{t.last_frame()}")
        self._project_changed()
        self._select("video", t.id)
        self.statusBar().showMessage(f"Restored '{t.label}' track.", 4000)

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
        self.tabs.setCurrentWidget(self.transcript)
        started = [False]

        def _begin():
            # Clear the old transcript only once the model has loaded and produced something, so a
            # failed model load (e.g. a size that is not bundled, offline) leaves the old text intact.
            if not started[0]:
                started[0] = True
                proj.transcript = []
                self.transcript.rebuild()

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
                _begin()
                proj.transcript.append(seg)
                self.transcript.append_segment(seg)
                self._update_enabled()

        def done(segments):
            if segments is None or self.project is not proj:
                return
            _begin()
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
        target = self._join_target(frame, bbox)
        if target is not None:
            # the box continues an existing track (drawn in one of its gaps, or after J): the new
            # tracking result is merged into that track instead of starting another one
            self.__dict__["_join_next"] = False
            self._run_tracking(frame, bbox, target.label, target, join_id=target.id)
            self.statusBar().showMessage(f"Adding this box to the selected '{target.label}' track (Ctrl+Z undoes the join).", 8000)
            return
        quick = bool(settings.get("quick_box", True)) and not (QApplication.keyboardModifiers() & Qt.ShiftModifier)
        if quick:
            # one gesture: drag a box, it is tracked forward and backward right away with the
            # label/shape/style used last time (Shift+drag opens the dialog instead)
            label = settings.get("last_box_label", "face") or "other"
            track = VideoTrack(label=label, style=as_enum(VideoStyle, settings.get("last_box_style")),
                               shape=as_enum(Shape, settings.get("last_box_shape")) or (Shape.ELLIPSE if label == "face" else Shape.RECT),
                               pad=settings["video_pad"], source="manual")
            self._run_tracking(frame, bbox, label, track)
            self.statusBar().showMessage(f"Tracking '{label}' forward and backward from here. "
                                         "Shift+drag to choose a different label or options; T re-tracks a selected box.", 8000)
            return
        dlg = NewBoxDialog(settings, self.current_time(), self.project.media.duration, self)
        if dlg.exec() != QDialog.Accepted:
            return
        label = dlg.label.currentText().strip() or "other"
        track = VideoTrack(label=label, style=as_enum(VideoStyle, dlg.style.currentData()),
                           shape=as_enum(Shape, dlg.shape.currentData()) or Shape.RECT,
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
            self._run_tracking(frame, bbox, label, track, step=dlg.track_step, backward=dlg.track_backward, end_frame=end_frame,
                               lost_limit_s=dlg.lost_limit_s, search_to_end=dlg.search_to_end)

    JOIN_GAP_S = 10.0   # a box drawn within this many seconds of a selected track's gap edge continues that track

    def join_next_box(self) -> None:
        """J: the next box drawn joins the selected track wherever it is drawn."""
        t = self._selected_track()
        if t is None:
            self.statusBar().showMessage("Select a video track first, then press J and draw the box that continues it.", 6000)
            return
        self.__dict__["_join_next"] = True
        self.statusBar().showMessage(f"Draw the box that continues the selected '{t.label}' track.", 8000)

    def _join_target(self, frame: int, bbox: BBox) -> Optional[VideoTrack]:
        """The selected track when the new box should continue it: J was pressed, or the box is
        drawn in a gap of the selected track within JOIN_GAP_S of one of its spans and is of a
        similar size to the box at that edge."""
        t = self._selected_track()
        if t is None or not t.spans:
            return None
        if self.__dict__.get("_join_next"):
            return t
        if t.bbox_at(frame) is not None:
            return None
        fps = self.fps or 30.0
        best = None
        for s in t.spans:
            if s.end_frame < frame:
                d, edge = frame - s.end_frame, s.bbox_at(s.end_frame)
            elif s.start_frame > frame:
                d, edge = s.start_frame - frame, s.bbox_at(s.start_frame)
            else:
                continue
            if edge is not None and (best is None or d < best[0]):
                best = (d, edge)
        if best is None or best[0] > self.JOIN_GAP_S * fps:
            return None
        sa, sb = max(bbox.w, bbox.h), max(best[1].w, best[1].h)
        return t if (sa > 0 and sb > 0 and 1 / 3 <= sa / sb <= 3) else None

    def _run_tracking(self, frame: int, bbox: BBox, label: str, proto: VideoTrack, replace_id: Optional[str] = None,
                      step: Optional[int] = None, backward: Optional[bool] = None, end_frame: Optional[int] = None,
                      lost_limit_s: Optional[float] = None, search_to_end: Optional[bool] = None,
                      join_id: Optional[str] = None) -> None:
        """Track in the background. The track is added immediately with the drawn box and
        fills in live as the tracker advances, so the UI stays usable. With `join_id` the
        result is merged into that existing track instead (a snapshot goes on the undo stack)."""
        proj = self.project
        path = proj.media.path
        det_label = label if label in ("face", "screen", "document", "phone", "person") else None
        face_conf, obj_conf, model, threads = settings["face_conf"], settings["detect_conf"], settings.get("detector_model", "yolox_tiny"), settings["cpu_threads"]
        face_model = settings.get("face_detector", "yunet")
        from videoredact.vision.tracker import TrackOptions
        opts = TrackOptions(engine=str(settings.get("track_engine", "anchored")),
                            step=int(step or settings.get("track_step", 2)),
                            backward=bool(settings.get("track_backward", True) if backward is None else backward),
                            end_frame=end_frame, tracker=settings.get("tracker", "vit"),
                            lost_limit_s=float(settings.get("track_lost_limit_s", 8.0) if lost_limit_s is None else lost_limit_s),
                            search_to_end=bool(settings.get("track_search_to_end", False) if search_to_end is None else search_to_end))
        joined = proj.get_track(join_id) if join_id else None
        if joined is not None:
            self._push_undo(copy.deepcopy(joined), kind="replaced")
            joined.add_keyframe(frame, bbox)
            live = joined
            self._project_changed()
        else:
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
            if joined is not None or self.project is not proj or proj.get_track(live.id) is not live:
                return
            live.spans = spans
            live.merge_spans()
            self.timeline.refresh()
            self.video.set_tracks(proj.video_tracks, proj.default_video_style)

        def done(track: VideoTrack):
            if track is None or self.project is not proj or proj.get_track(live.id) is not live:
                return
            if joined is not None:
                from videoredact.vision.tracker import bridge_spans
                before = live.total_frames()
                live.spans = live.spans + track.spans
                live.merge_spans()
                bridge_spans(live, int(2.0 * (self.fps or 30.0)))
                live.note = (live.note + "; " if live.note else "") + f"joined a box drawn at frame {frame}"
                proj.log("track_joined", f"{label} +{live.total_frames() - before} frames from a box at frame {frame}")
                self._project_changed()
                self._select("video", live.id)
                self.statusBar().showMessage(f"Joined: '{label}' track now covers {live.total_frames()} frames in "
                                             f"{len(live.spans)} span(s) (Ctrl+Z undoes the join)", 8000)
                return
            live.spans = track.spans
            live.note = f"tracked from frame {frame}; {len(track.spans)} visible span(s); every {opts.step} frame(s)"
            proj.log("tracked", f"{label} {live.total_frames()} frames in {len(live.spans)} span(s)")
            self._project_changed()
            self.statusBar().showMessage(f"Tracked '{label}': {live.total_frames()} frames in {len(live.spans)} span(s)", 8000)

        def on_cancel():
            if self.project is proj and joined is None:
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

    def _selected_track(self) -> Optional[VideoTrack]:
        kind, rid = self.panel.current()
        tid = rid if kind == "video" else self.video.selected_track_id
        return self.project.get_track(tid) if (self.project and tid) else None

    def trim_selected(self, which: str) -> None:
        t = self._selected_track()
        if not t:
            return
        f = self.current_frame()
        (t.trim_start if which == "start" else t.trim_end)(f)
        if not t.spans:
            self.project.remove_video_track(t.id)
        else:
            self.project.log("track_trimmed", f"{t.label} {which} -> frame {f}")
        self._project_changed()

    def extend_selected(self) -> None:
        t = self._selected_track()
        if not t:
            return
        f = self.current_frame()
        t.extend_to(f)
        self.project.log("track_extended", f"{t.label} -> frame {f}")
        self._project_changed()

    def split_selected(self) -> None:
        """X: cut the selected track at the playhead into two tracks (delete the half that drifted)."""
        t = self._selected_track()
        if not t:
            return
        f = self.current_frame()
        tail = t.split_at(f)
        if tail is None:
            self.statusBar().showMessage("Move the playhead inside the track (not at its first frame) to split it.", 4000)
            return
        self.project.add_video_track(tail)
        self.project.log("track_split", f"{t.label} at frame {f}")
        self._project_changed()
        self._select("video", tail.id)

    def toggle_selected(self) -> None:
        """E: keep the track in the project but do not render it (or turn it back on)."""
        t = self._selected_track()
        if not t:
            return
        t.enabled = not t.enabled
        self.project.log(f"video_redaction_{'enabled' if t.enabled else 'disabled'}", t.id)
        self._project_changed()
        self.statusBar().showMessage(f"'{t.label}' track {'enabled' if t.enabled else 'disabled'}.", 3000)

    def toggle_quick_box(self) -> None:
        settings["quick_box"] = self.a_quick.isChecked()
        settings.save()
        self.statusBar().showMessage("New boxes are tracked immediately with the last settings; Shift+drag asks first."
                                     if self.a_quick.isChecked() else "Each new box asks what it is before tracking.", 6000)

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
        settings["detect_carry"] = dlg.carry.isChecked()
        settings.save()
        path = self.project.media.path
        from videoredact.vision.auto_detect import AutoDetectOptions, run_auto_detect
        opts = AutoDetectOptions(labels=labels, stride=dlg.stride.value(), face_conf=settings["face_conf"],
                                 obj_conf=settings["detect_conf"], obj_model=settings.get("detector_model", "yolox_tiny"),
                                 threads=settings["cpu_threads"])
        replace = dlg.replace.isChecked()

        opts.apply_profile(dlg.precise.isChecked())
        settings["detect_precise"] = dlg.precise.isChecked()
        opts.face_model = settings.get("face_detector", "yunet")
        opts.carry = dlg.carry.isChecked()
        opts.tiles = dlg.tiles.isChecked()
        opts.multiscale = dlg.ms.isChecked()
        settings["detect_multiscale"] = opts.multiscale
        opts.head_fallback = dlg.head.isChecked()
        opts.face_pad = float(settings.get("face_pad", 0.25))
        opts.assoc = settings.get("assoc", "botsort")
        opts.lead_in = int(settings.get("lead_frames", 15))
        opts.lead_out = int(settings.get("lead_frames", 15))
        settings["detect_tiles"] = opts.tiles
        settings["detect_head"] = opts.head_fallback
        settings.save()
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
    def _jobs_busy(self, what: str) -> bool:
        """True (after telling the user) when a tracking/detection/transcription job is still running,
        so an export or batch would ship a half-finished redaction."""
        if self.jobs.running:
            QMessageBox.warning(self, what, f"{self.jobs.running} background job(s) are still running "
                                "(tracking, detection or transcription). Wait for them to finish, or cancel them in "
                                "the Jobs panel, before starting this.")
            return True
        return False

    def export_media(self) -> None:
        if not self.project:
            return
        self.player.pause()
        if self._jobs_busy("Export"):
            return
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
        from videoredact.core.export import _report_paths
        out_p = Path(opts.output_path)
        existing = [p for p in (out_p, *_report_paths(out_p), Path(str(out_p.with_suffix("")) + "_audit.json")) if p.exists()]
        if existing:
            names = "\n".join(f"  {p.name}" for p in existing)
            if QMessageBox.question(self, "Replace files?",
                                    f"These files already exist in {out_p.parent} and will be replaced:\n\n{names}\n\n"
                                    "Replace them? (Choose No to pick another name.)",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
        settings["export_crf"], settings["export_preset"] = opts.crf, opts.preset
        settings["verify_export"] = bool(opts.verify_labels)
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
            if res.get("clip"):
                a, b = res["clip"]
                lines.append(f"Clip: {fmt_time(a)} - {fmt_time(b) if b is not None else 'end'} (a slate is NOT inserted; note the range in your release letter)")
            if "report_pdf" in res:
                lines.append(f"\nReport: {res['report_pdf']}\n        {res['report_csv']}")
            if "project" in res:
                lines.append(f"Project: {res['project']}")
            if res.get("audit_json"):
                lines.append(f"Audit manifest: {res['audit_json']}\n   SHA-256 {res['manifest_sha256'][:32]}…")
            if res.get("source_unchanged") is False:
                lines.append("\nWARNING: the source file's SHA-256 at export differs from the hash taken when it was opened. "
                             "The file changed on disk; check chain of custody before releasing.")
            lines.append(f"\nTook {res['elapsed_s']:.0f} s.")
            vr = res.get("verify")
            icon = QMessageBox.Information
            if vr is not None:
                if vr.uncovered:
                    icon = QMessageBox.Warning
                    lines.append(f"\nVERIFICATION: {len(vr.uncovered)} face detection(s) in the OUTPUT are not covered "
                                 f"(checked {vr.frames_checked} frames). Review these times and add boxes:")
                    for u in vr.uncovered[:12]:
                        lines.append(f"   {fmt_time(u.time)}  {u.label} (conf {u.score:.2f}, {u.covered:.0%} covered)")
                    if len(vr.uncovered) > 12:
                        lines.append(f"   … {len(vr.uncovered) - 12} more in {res.get('verification_csv', '')}")
                else:
                    lines.append(f"\nVerification: no visible faces found in the output ({vr.frames_checked} frames checked).")
            box = QMessageBox(icon, "Export complete", "\n".join(lines), parent=self)
            if vr is not None and vr.uncovered:
                jump = box.addButton("Jump to first", QMessageBox.ActionRole)
            else:
                jump = None
            open_btn = box.addButton("Open folder", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Ok)
            box.exec()
            if box.clickedButton() == open_btn:
                os.startfile(str(Path(res["output"]).parent))  # type: ignore[attr-defined]
            elif jump is not None and box.clickedButton() == jump:
                self.seek(vr.uncovered[0].time)

        self.jobs.run("Exporting redacted media", job, done, on_cancel=on_cancel)

    # --------------------------------------------------------------- batch
    def batch_process(self) -> None:
        if self._jobs_busy("Batch"):
            return
        dlg = BatchDialog(settings, self)
        if dlg.exec() != QDialog.Accepted or not dlg.files():
            return
        from videoredact.core.batch import BatchOptions, run_batch
        settings["batch_out_dir"] = dlg.out.text().strip()
        settings["batch_export"] = dlg.export.isChecked()
        settings.save()
        opts = BatchOptions(files=dlg.files(), out_dir=dlg.out.text().strip() or None, detect_labels=dlg.labels(),
                            stride=int(settings.get("detect_stride", 5)), export=dlg.export.isChecked(),
                            verify=bool(settings.get("verify_export", True)), author=settings.get("author", ""),
                            default_video_style=settings.get("default_video_style", "black"),
                            default_audio_style=settings.get("default_audio_style", "beep"),
                            crf=int(settings.get("export_crf", 23)), preset=settings.get("export_preset", "veryfast"),
                            detect_kwargs=dict(face_conf=settings["face_conf"], obj_conf=settings["detect_conf"],
                                               obj_model=settings.get("detector_model", "yolox_tiny"),
                                               face_model=settings.get("face_detector", "yunet"),
                                               carry=bool(settings.get("detect_carry", True)),
                                               tiles=bool(settings.get("detect_tiles", False)),
                                               multiscale=bool(settings.get("detect_multiscale", True)),
                                               head_fallback=bool(settings.get("detect_head", True)),
                                               face_pad=float(settings.get("face_pad", 0.25)),
                                               assoc=settings.get("assoc", "botsort"),
                                               lead_in=int(settings.get("lead_frames", 15)),
                                               lead_out=int(settings.get("lead_frames", 15))))

        def job(prog, cancel, partial):
            return run_batch(opts, prog, cancel)

        def done(results):
            if results is None:
                return
            ok = [r for r in results if r.ok]
            bad = [r for r in results if not r.ok]
            lines = [f"{len(ok)} of {len(results)} file(s) processed."]
            for r in ok[:15]:
                lines.append(f"  {Path(r.file).name}: {r.tracks} tracks, {r.seconds:.0f} s"
                             + (f", {r.uncovered} uncovered in verification" if r.uncovered is not None else ""))
            for r in bad[:5]:
                lines.append(f"  FAILED {Path(r.file).name}: {r.error.splitlines()[0][:120]}")
            QMessageBox.information(self, "Batch complete", "\n".join(lines))

        self.jobs.run(f"Batch: {len(opts.files)} file(s)", job, done)

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
            names = reg.required_models()   # optional models without a download source are skipped
            for i, n in enumerate(names):
                if cancel():
                    return None
                prog(i / len(names), f"Checking {n}…")
                reg.ensure_model(n, lambda p, m: prog((i + p) / len(names), m))
            return True

        from videoredact.paths import model_search_dirs
        where = "\n".join(str(d) for d in model_search_dirs() if d.exists())
        optional = [n for n in reg.MODELS if reg.is_optional(n) and not reg.is_available(n)]
        note = f"\n\nOptional, not installed (not needed): {', '.join(optional)}" if optional else ""
        run_task(self, "Checking detection models…", job,
                 lambda ok: ok and QMessageBox.information(self, "Models", f"All detection models are present in:\n{where}{note}"))

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
                # background thread + short timeout: the UI never waits on the network; a
                # failure at startup is silent (offline machines), a manual check reports it
                try:
                    self_.result.emit(updater.fetch_latest(timeout=updater.DEFAULT_TIMEOUT_S), "")
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
                          + "Nothing is installed automatically. Open the release page to download the "
                            "installer, then run it (an administrator may be required on managed PCs).",
                          parent=self)
        box.setInformativeText(info.page)
        b_open = box.addButton("Open download page", QMessageBox.AcceptRole)
        box.addButton("Later", QMessageBox.RejectRole)
        b_skip = box.addButton("Skip this version", QMessageBox.DestructiveRole)
        box.setDefaultButton(b_open)
        box.exec()
        if box.clickedButton() == b_skip:
            settings["skip_version"] = info.version
            settings.save()
        elif box.clickedButton() == b_open:
            if not QDesktopServices.openUrl(QUrl(info.page)):
                updater.open_release_page(info)

    def open_log_folder(self) -> None:
        from videoredact.paths import log_dir
        os.startfile(str(log_dir()))  # type: ignore[attr-defined]

    def quick_guide(self) -> None:
        QMessageBox.information(self, "Quick guide", """
<b>1. Open</b> a video or audio file (File ▸ Open media).<br>
<b>2. Audio:</b> Audio ▸ Transcribe. In the Transcript tab select words, right-click ▸ Redact,
or ▸ Redact every occurrence. Use the search box to find a word everywhere.
Audio ▸ Suggest sensitive speech finds phone numbers, names after cues, addresses, dates of birth.<br>
<b>3. Video:</b> Video ▸ Auto-detect finds faces, screens, documents and phones. By default it keeps only confident
detections (a face must be seen 3+ times at 70 %+ confidence); untick that option to find everything and delete the
false hits yourself. Drag a box on the video around anything else (a license plate, a tattoo, a notepad): it is tracked
forward and backward immediately with the last label and style, and stops once the object has been out of view for 8 s.
Shift+drag asks for the label, style and tracking options first. If a track is selected and you draw a box in one of
its gaps (within 10 s of where it stopped), the box continues that track instead of starting a new one; press J first
to force that from anywhere.<br>
<b>4. Review:</b> Play with <i>Preview redactions</i> on: you see the boxes and hear the beep/silence exactly as exported.
N / Shift+N step through the redactions, Delete removes one (Ctrl+Z brings it back), E disables it without deleting,
X splits a track at the playhead, [ ] trim it, = extends it, T re-tracks a selected box from the playhead.
Drag a box to fix its position: that adds a keyframe.<br>
<b>5. Export</b> (Ctrl+E): writes the redacted file plus a PDF/CSV report with SHA-256 hashes and the project file.<br><br>
<b>Keys:</b> Space play/pause · ←/→ frame step (Shift: 1 s) · N / Shift+N next/previous redaction · F / Shift+F next/previous
flagged track · Delete remove selected · Ctrl+Z undo delete · E enable/disable · X split at playhead · [ ] trim start/end to
playhead · = extend to playhead · T track selected box · J join the next box to the selected track · Ctrl+R redact audio range at playhead · Ctrl+F find word ·
Ctrl+E export · Ctrl+B batch · Ctrl+wheel zoom timeline · double-click video to fit.
""")

    def about(self) -> None:
        QMessageBox.about(self, "About VideoRedact",
                          f"<b>VideoRedact {__version__}</b><br>Free, offline video and audio redaction for law enforcement.<br>"
                          "MIT License. Runs entirely on this computer; no media data leaves the machine "
                          "(the only network access is the optional update check against GitHub).<br><br>"
                          "Components: FFmpeg (LGPL build), OpenCV, ONNX Runtime, faster-whisper, YuNet, YOLOX, Qt/PySide6 (LGPL).")

    _TRANSPORT_KEYS = (Qt.Key_Space, Qt.Key_Left, Qt.Key_Right)

    def _transport_key(self, event) -> bool:
        """Space = play/pause, Left/Right = frame step (Shift: one second). Returns True if handled."""
        k = event.key()
        if self.project is None or k not in self._TRANSPORT_KEYS:
            return False
        fw = QApplication.focusWidget()
        # never steal keys from something the user is typing in or navigating with arrows
        from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QComboBox, QLineEdit, QPlainTextEdit,
                                       QTextEdit)
        if isinstance(fw, (QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit)):
            return False
        if isinstance(fw, QTextEdit) and not fw.isReadOnly():
            return False
        if isinstance(fw, QAbstractItemView) and k != Qt.Key_Space:
            return False
        if k == Qt.Key_Space:
            self.toggle_play()
        elif k == Qt.Key_Left:
            self.step_frames(-(int(self.fps) if event.modifiers() & Qt.ShiftModifier else 1))
        else:
            self.step_frames(int(self.fps) if event.modifiers() & Qt.ShiftModifier else 1)
        return True

    def eventFilter(self, obj, event):
        # The video view, timeline and transcript are scroll areas / text views that consume arrow
        # keys themselves, so keyPressEvent on the window never saw them after a click there.
        # An application-level filter handles the transport keys first (only while this window is
        # active, so dialogs keep their own keyboard handling).
        if event.type() == QEvent.KeyPress and QApplication.activeWindow() is self and self._transport_key(event):
            return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if not self._transport_key(event):
            super().keyPressEvent(event)
