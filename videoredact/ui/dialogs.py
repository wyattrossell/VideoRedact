"""Dialogs: settings, export, auto-detect options, new-box action, time range."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton,
                               QSpinBox, QVBoxLayout, QWidget, QTextEdit)

from videoredact.audio.transcribe import MODEL_SIZES
from videoredact.core.model import AudioStyle, Project, Shape, VideoStyle
from videoredact.paths import Settings, models_dir
from videoredact.vision.detector import AUTO_LABELS, LABEL_DESCRIPTIONS
from videoredact.vision import models as model_registry
from .panels import AUDIO_STYLE_NAMES, VIDEO_STYLE_NAMES

BOX_LABELS = ["face", "license_plate", "screen", "document", "phone", "person", "tattoo", "other"]


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, project: Optional[Project], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings
        self.project = project
        lay = QVBoxLayout(self)

        g1 = QGroupBox("Operator / case")
        f1 = QFormLayout(g1)
        self.author = QLineEdit(settings["author"])
        self.case = QLineEdit(project.case_number if project else "")
        self.notes = QTextEdit(project.notes if project else "")
        self.notes.setMaximumHeight(60)
        f1.addRow("Your name (appears in reports)", self.author)
        f1.addRow("Case number (this project)", self.case)
        f1.addRow("Notes (this project)", self.notes)
        lay.addWidget(g1)

        g2 = QGroupBox("Redaction defaults")
        f2 = QFormLayout(g2)
        self.a_style = QComboBox()
        for k, v in AUDIO_STYLE_NAMES.items():
            self.a_style.addItem(v, k.value)
        self.a_style.setCurrentIndex(list(AUDIO_STYLE_NAMES).index(AudioStyle(settings["default_audio_style"])))
        self.v_style = QComboBox()
        for k, v in VIDEO_STYLE_NAMES.items():
            self.v_style.addItem(v, k.value)
        self.v_style.setCurrentIndex(list(VIDEO_STYLE_NAMES).index(VideoStyle(settings["default_video_style"])))
        self.beep = QDoubleSpinBox()
        self.beep.setRange(200, 4000)
        self.beep.setValue(settings["beep_frequency"])
        self.beep.setSuffix(" Hz")
        self.apad = QDoubleSpinBox()
        self.apad.setRange(0, 1)
        self.apad.setSingleStep(0.01)
        self.apad.setDecimals(2)
        self.apad.setValue(settings["audio_pad_s"])
        self.apad.setSuffix(" s")
        self.vpad = QSpinBox()
        self.vpad.setRange(0, 100)
        self.vpad.setValue(int(settings["video_pad"] * 100))
        self.vpad.setSuffix(" %")
        f2.addRow("Audio style", self.a_style)
        f2.addRow("Beep frequency", self.beep)
        f2.addRow("Extra audio padding each side", self.apad)
        f2.addRow("Video style", self.v_style)
        f2.addRow("Box margin (new boxes)", self.vpad)
        lay.addWidget(g2)

        g3 = QGroupBox("Speech recognition (runs offline on CPU)")
        f3 = QFormLayout(g3)
        self.model = QComboBox()
        self.model.addItems(MODEL_SIZES)
        self.model.setCurrentText(settings["whisper_model"])
        self.model.setToolTip("tiny/base: fast, rough. small: good balance. medium: slow, accurate. "
                              "large-v3-turbo: most accurate, needs ~6 GB RAM.")
        self.lang = QLineEdit(settings["language"])
        self.lang.setPlaceholderText("auto-detect  (or: en, es, …)")
        self.vad = QCheckBox("Skip silence with a voice detector first (faster, but misses quiet speech in noisy audio)")
        self.vad.setChecked(bool(settings.get("vad", False)))
        self.threads = QSpinBox()
        self.threads.setRange(0, 64)
        self.threads.setValue(settings["cpu_threads"])
        self.threads.setSpecialValueText("auto")
        f3.addRow("Whisper model", self.model)
        f3.addRow("Language", self.lang)
        f3.addRow("", self.vad)
        f3.addRow("CPU threads", self.threads)
        lay.addWidget(g3)

        g4 = QGroupBox("Object detection")
        f4 = QFormLayout(g4)
        self.det_model = QComboBox()
        self.det_model.addItem("YOLOX-s (accurate, ~0.15 s/frame)", "yolox_s")
        self.det_model.addItem("YOLOX-tiny (fast, ~0.05 s/frame)", "yolox_tiny")
        self.det_model.setCurrentIndex(0 if settings.get("detector_model", "yolox_tiny") == "yolox_s" else 1)
        self.face_model = QComboBox()
        self.face_model.addItem("YuNet (fast)", "yunet")
        self.face_model.addItem("CenterFace (better on small / side faces, slower)", "centerface")
        self.face_model.setCurrentIndex(1 if settings.get("face_detector", "yunet") == "centerface" else 0)
        self.stride = QSpinBox()
        self.stride.setRange(1, 30)
        self.stride.setValue(settings["detect_stride"])
        self.stride.setToolTip("Run detectors every N frames; boxes are interpolated in between.")
        self.dconf = QDoubleSpinBox()
        self.dconf.setRange(0.05, 0.95)
        self.dconf.setSingleStep(0.05)
        self.dconf.setValue(settings["detect_conf"])
        self.fconf = QDoubleSpinBox()
        self.fconf.setRange(0.05, 0.99)
        self.fconf.setSingleStep(0.05)
        self.fconf.setValue(settings["face_conf"])
        f4.addRow("Object model", self.det_model)
        f4.addRow("Face model", self.face_model)
        f4.addRow("Detect every N frames", self.stride)
        f4.addRow("Object confidence", self.dconf)
        f4.addRow("Face confidence", self.fconf)
        f4.addRow("Models folder", QLabel(str(models_dir())))
        lay.addWidget(g4)

        g5 = QGroupBox("Updates")
        f5 = QFormLayout(g5)
        self.updates = QCheckBox("Check GitHub for a newer version at startup")
        self.updates.setChecked(bool(settings.get("check_updates", True)))
        self.updates.setToolTip("Only the version number of the latest release is requested from GitHub. "
                                "No media or project data is ever sent.")
        f5.addRow(self.updates)
        lay.addWidget(g5)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def apply(self) -> None:
        s = self.settings
        s["author"] = self.author.text().strip()
        s["default_audio_style"] = self.a_style.currentData()
        s["default_video_style"] = self.v_style.currentData()
        s["beep_frequency"] = self.beep.value()
        s["audio_pad_s"] = self.apad.value()
        s["video_pad"] = self.vpad.value() / 100
        s["whisper_model"] = self.model.currentText()
        s["language"] = self.lang.text().strip()
        s["cpu_threads"] = self.threads.value()
        s["detector_model"] = self.det_model.currentData()
        s["face_detector"] = self.face_model.currentData()
        s["vad"] = self.vad.isChecked()
        s["detect_stride"] = self.stride.value()
        s["detect_conf"] = self.dconf.value()
        s["face_conf"] = self.fconf.value()
        s["check_updates"] = self.updates.isChecked()
        if self.updates.isChecked():
            s["skip_version"] = ""
        s.save()
        if self.project:
            self.project.author = s["author"]
            self.project.case_number = self.case.text().strip()
            self.project.notes = self.notes.toPlainText().strip()
            self.project.default_audio_style = AudioStyle(s["default_audio_style"])
            self.project.default_video_style = VideoStyle(s["default_video_style"])
            self.project.beep_frequency = s["beep_frequency"]


class AutoDetectDialog(QDialog):
    def __init__(self, settings: Settings, has_selection_range: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Automatic detection")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Find and track these kinds of objects through the whole video.\n"
                             "Review results in the Redactions panel afterwards; delete any false hits."))
        self.checks: dict[str, QCheckBox] = {}
        for lbl in AUTO_LABELS:
            cb = QCheckBox(LABEL_DESCRIPTIONS[lbl])
            cb.setChecked(lbl in settings["detect_classes"])
            self.checks[lbl] = cb
            lay.addWidget(cb)
        note = QLabel("License plates: no permissively-licensed CPU model is available yet. "
                      "Draw a box around a plate and use Track object instead.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray")
        lay.addWidget(note)
        f = QFormLayout()
        self.stride = QSpinBox()
        self.stride.setRange(1, 30)
        self.stride.setValue(settings["detect_stride"])
        f.addRow("Detect every N frames", self.stride)
        self.replace = QCheckBox("Remove previous automatic results first")
        self.replace.setChecked(True)
        lay.addLayout(f)
        lay.addWidget(self.replace)
        est = QLabel("")
        est.setStyleSheet("color: gray")
        self.est = est
        lay.addWidget(est)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Run detection")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def labels(self) -> list[str]:
        return [k for k, cb in self.checks.items() if cb.isChecked()]

    def set_estimate(self, frames: int, fps: float) -> None:
        n = frames / max(1, self.stride.value())
        secs = n * 0.17
        self.est.setText(f"About {n:,.0f} detector passes; roughly {secs / 60:.0f} min on an 8-core CPU (YOLOX-s). "
                         f"Faces only is several times faster.")


class ExportDialog(QDialog):
    def __init__(self, project: Project, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export redacted media")
        self.project = project
        lay = QVBoxLayout(self)
        f = QFormLayout()
        src = Path(project.media.path)
        default_ext = ".mp4" if project.media.has_video else (src.suffix if src.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"} else ".m4a")
        self.path = QLineEdit(str(src.with_name(src.stem + "_REDACTED" + default_ext)))
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.path, 1)
        row.addWidget(b)
        w = QWidget()
        w.setLayout(row)
        f.addRow("Output file", w)
        if project.media.has_video:
            self.quality = QComboBox()
            self.quality.addItem("High (CRF 18, visually lossless)", 18)
            self.quality.addItem("Medium (CRF 23, smaller file)", 23)
            self.quality.addItem("Low (CRF 28)", 28)
            self.quality.setCurrentIndex([18, 23, 28].index(settings.get("export_crf", 18)) if settings.get("export_crf", 18) in (18, 23, 28) else 0)
            f.addRow("Video quality", self.quality)
            self.preset = QComboBox()
            self.preset.addItems(["ultrafast", "veryfast", "medium", "slow"])
            self.preset.setCurrentText(settings.get("export_preset", "veryfast"))
            f.addRow("Encoder speed", self.preset)
        self.report = QCheckBox("Write redaction report (PDF + CSV) next to the output")
        self.report.setChecked(True)
        self.proj = QCheckBox("Write project file (.vrproj) next to the output")
        self.proj.setChecked(True)
        lay.addLayout(f)
        lay.addWidget(self.report)
        lay.addWidget(self.proj)
        n_a = sum(1 for r in project.audio_redactions if r.enabled)
        n_v = sum(1 for t in project.video_tracks if t.enabled)
        lay.addWidget(QLabel(f"{n_a} audio redaction(s) and {n_v} video region(s) will be applied. "
                             f"The original file is not modified."))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Export")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(620, self.sizeHint().height())

    def _browse(self) -> None:
        p, _ = QFileDialog.getSaveFileName(self, "Save redacted media", self.path.text(),
                                           "Video (*.mp4);;Audio (*.m4a *.mp3 *.wav);;All files (*)")
        if p:
            self.path.setText(p)

    def options(self):
        from videoredact.core.export import ExportOptions
        o = ExportOptions(self.path.text().strip())
        if self.project.media.has_video:
            o.crf = self.quality.currentData()
            o.preset = self.preset.currentText()
        o.write_report = self.report.isChecked()
        o.write_project = self.proj.isChecked()
        return o


class NewBoxDialog(QDialog):
    """Asked after the user draws a box: what is it and what should we do?"""

    def __init__(self, settings: Settings, cur_time: float, duration: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New redaction box")
        lay = QVBoxLayout(self)
        f = QFormLayout()
        self.label = QComboBox()
        self.label.setEditable(True)
        self.label.addItems(BOX_LABELS)
        self.label.setCurrentText(settings.get("last_box_label", "face"))
        f.addRow("What is it?", self.label)
        self.shape = QComboBox()
        self.shape.addItem("Rectangle", Shape.RECT.value)
        self.shape.addItem("Ellipse", Shape.ELLIPSE.value)
        f.addRow("Shape", self.shape)
        self.style = QComboBox()
        self.style.addItem("Project default", None)
        for k, v in VIDEO_STYLE_NAMES.items():
            self.style.addItem(v, k.value)
        f.addRow("Style", self.style)
        lay.addLayout(f)
        g = QGroupBox("Then…")
        gl = QVBoxLayout(g)
        self.r_track = QRadioButton("Track this object automatically (forward and backward, re-acquire if it leaves the frame)")
        self.r_track.setChecked(True)
        self.r_static = QRadioButton("Keep the box fixed over a time range")
        self.r_key = QRadioButton("Just this frame (I will add keyframes manually)")
        gl.addWidget(self.r_track)
        gl.addWidget(self.r_static)
        rng = QHBoxLayout()
        self.t0 = QDoubleSpinBox()
        self.t0.setRange(0, duration)
        self.t0.setDecimals(2)
        self.t0.setValue(cur_time)
        self.t1 = QDoubleSpinBox()
        self.t1.setRange(0, duration)
        self.t1.setDecimals(2)
        self.t1.setValue(duration)
        rng.addSpacing(24)
        rng.addWidget(QLabel("from"))
        rng.addWidget(self.t0)
        rng.addWidget(QLabel("to"))
        rng.addWidget(self.t1)
        rng.addWidget(QLabel("seconds"))
        rng.addStretch(1)
        gl.addLayout(rng)
        gl.addWidget(self.r_key)
        lay.addWidget(g)

        g2 = QGroupBox("Tracking options")
        f2 = QFormLayout(g2)
        self.step = QComboBox()
        self.step.addItem("Every 2nd frame (recommended)", 2)
        self.step.addItem("Every frame (slower, most precise)", 1)
        self.step.addItem("Every 3rd frame (fastest)", 3)
        cur_step = int(settings.get("track_step", 2))
        self.step.setCurrentIndex({2: 0, 1: 1, 3: 2}.get(cur_step, 0))
        f2.addRow("Analyse", self.step)
        self.backward = QCheckBox("Also look backward from here (covers the object before this frame)")
        self.backward.setChecked(bool(settings.get("track_backward", True)))
        f2.addRow("", self.backward)
        self.stop_at = QDoubleSpinBox()
        self.stop_at.setRange(0, duration)
        self.stop_at.setDecimals(1)
        self.stop_at.setValue(duration)
        self.stop_at.setSuffix(" s")
        self.stop_at.setToolTip("Stop tracking at this time. Shorter ranges finish sooner.")
        f2.addRow("Stop at", self.stop_at)
        lay.addWidget(g2)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.settings = settings

    @property
    def track_step(self) -> int:
        return int(self.step.currentData())

    @property
    def track_backward(self) -> bool:
        return self.backward.isChecked()

    @property
    def track_end_time(self) -> float:
        return float(self.stop_at.value())

    def mode(self) -> str:
        if self.r_track.isChecked():
            return "track"
        if self.r_static.isChecked():
            return "static"
        return "keyframe"

    def accept(self) -> None:
        self.settings["last_box_label"] = self.label.currentText().strip() or "other"
        self.settings["track_step"] = self.track_step
        self.settings["track_backward"] = self.track_backward
        self.settings.save()
        super().accept()
