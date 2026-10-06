"""Side panels: redaction list and PII suggestion dialog."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget, QCheckBox)

from videoredact.audio.pii import Suggestion
from videoredact.core.model import AudioStyle, Project, Shape, VideoStyle

AUDIO_STYLE_NAMES = {AudioStyle.BEEP: "Beep", AudioStyle.SILENCE: "Silence", AudioStyle.LOW_TONE: "Low tone",
                     AudioStyle.NOISE: "Noise"}
VIDEO_STYLE_NAMES = {VideoStyle.BLACK: "Black box", VideoStyle.BLUR: "Blur", VideoStyle.PIXELATE: "Pixelate"}


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


class RedactionPanel(QWidget):
    """Tree of all audio redactions and video tracks with per-item controls."""
    seekRequested = Signal(float)
    selectionChanged = Signal(str, str)   # kind ('audio'|'video'|''), id
    changed = Signal()                    # project modified (enable/style)
    deleteRequested = Signal(str, str)    # kind, id
    trackRequested = Signal(str)          # video track id: (re)track from its first keyframe

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[Project] = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["On", "What", "Start", "End", "Style"])
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setRootIsDecorated(True)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.itemSelectionChanged.connect(self._on_select)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_double)
        lay.addWidget(self.tree, 1)
        btns = QHBoxLayout()
        self.btn_del = QPushButton("Delete")
        self.btn_del.clicked.connect(self._delete)
        self.btn_track = QPushButton("Track object")
        self.btn_track.setToolTip("Follow the selected box through the video (forward and backward) with automatic re-acquisition")
        self.btn_track.clicked.connect(self._track)
        btns.addWidget(self.btn_del)
        btns.addWidget(self.btn_track)
        btns.addStretch(1)
        lay.addLayout(btns)
        self.summary = QLabel("")
        self.summary.setStyleSheet("color: gray")
        lay.addWidget(self.summary)
        self._building = False
        self.audio_root = None
        self.video_root = None

    def set_project(self, p: Optional[Project]) -> None:
        self.project = p
        self.rebuild()

    def rebuild(self, keep_selection: Optional[str] = None) -> None:
        self._building = True
        self.tree.clear()
        self.audio_root = QTreeWidgetItem(["", "Audio redactions", "", "", ""])
        self.video_root = QTreeWidgetItem(["", "Video redactions", "", "", ""])
        self.tree.addTopLevelItem(self.audio_root)
        self.tree.addTopLevelItem(self.video_root)
        sel_item = None
        if self.project:
            p = self.project
            for r in p.audio_redactions:
                it = QTreeWidgetItem(["", r.text or f"range ({r.source})", _fmt(r.start), _fmt(r.end), ""])
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Checked if r.enabled else Qt.Unchecked)
                it.setData(0, Qt.UserRole, ("audio", r.id))
                it.setToolTip(1, f"{r.text}\nreason: {r.reason}\nsource: {r.source}\nby {r.created_by} at {r.created_at}")
                self.audio_root.addChild(it)
                combo = self._style_combo(AUDIO_STYLE_NAMES, r.style, p.default_audio_style,
                                          lambda v, rr=r: self._set_audio_style(rr, v))
                self.tree.setItemWidget(it, 4, combo)
                if r.id == keep_selection:
                    sel_item = it
            fps = p.media.fps or 30.0
            for t in p.video_tracks:
                it = QTreeWidgetItem(["", f"{t.label}  ({t.source}, {len(t.spans)} span{'s' if len(t.spans) != 1 else ''})",
                                      _fmt(t.first_frame() / fps), _fmt((t.last_frame() + 1) / fps), ""])
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Checked if t.enabled else Qt.Unchecked)
                it.setData(0, Qt.UserRole, ("video", t.id))
                it.setToolTip(1, f"{t.label}\n{t.note}\nshape: {t.shape.value}, pad: {t.pad:.0%}\nby {t.created_by} at {t.created_at}")
                self.video_root.addChild(it)
                combo = self._style_combo(VIDEO_STYLE_NAMES, t.style, p.default_video_style,
                                          lambda v, tt=t: self._set_video_style(tt, v))
                self.tree.setItemWidget(it, 4, combo)
                if t.id == keep_selection:
                    sel_item = it
            self.summary.setText(f"{len(p.audio_redactions)} audio, {len(p.video_tracks)} video")
        self.audio_root.setExpanded(True)
        self.video_root.setExpanded(True)
        self._building = False
        if sel_item:
            self.tree.setCurrentItem(sel_item)

    def _style_combo(self, names: dict, current, default, cb) -> QComboBox:
        c = QComboBox()
        c.addItem(f"Default ({names[default]})", None)
        for k, v in names.items():
            c.addItem(v, k)
        idx = 0 if current is None else list(names).index(current) + 1
        c.setCurrentIndex(idx)
        c.currentIndexChanged.connect(lambda i, cc=c: cb(cc.itemData(i)))
        c.setMaximumWidth(150)
        return c

    def _set_audio_style(self, r, v) -> None:
        r.style = v
        self.project.log("audio_style_changed", f"{r.id} -> {v.value if v else 'default'}")
        self.changed.emit()

    def _set_video_style(self, t, v) -> None:
        t.style = v
        self.project.log("video_style_changed", f"{t.id} -> {v.value if v else 'default'}")
        self.changed.emit()

    def _on_item_changed(self, item, col) -> None:
        if self._building or col != 0 or not self.project:
            return
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        kind, rid = data
        on = item.checkState(0) == Qt.Checked
        if kind == "audio":
            for r in self.project.audio_redactions:
                if r.id == rid:
                    r.enabled = on
        else:
            t = self.project.get_track(rid)
            if t:
                t.enabled = on
        self.project.log(f"{kind}_redaction_{'enabled' if on else 'disabled'}", rid)
        self.changed.emit()

    def _on_select(self) -> None:
        if self._building:
            return
        items = self.tree.selectedItems()
        data = items[0].data(0, Qt.UserRole) if items else None
        if data:
            self.selectionChanged.emit(data[0], data[1])
        else:
            self.selectionChanged.emit("", "")

    def _on_double(self, item, col) -> None:
        data = item.data(0, Qt.UserRole)
        if not data or not self.project:
            return
        kind, rid = data
        if kind == "audio":
            r = next((x for x in self.project.audio_redactions if x.id == rid), None)
            if r:
                self.seekRequested.emit(r.start)
        else:
            t = self.project.get_track(rid)
            if t:
                self.seekRequested.emit(t.first_frame() / (self.project.media.fps or 30.0))

    def current(self) -> tuple[str, str]:
        items = self.tree.selectedItems()
        data = items[0].data(0, Qt.UserRole) if items else None
        return data if data else ("", "")

    def select(self, kind: str, rid: str) -> None:
        root = self.audio_root if kind == "audio" else self.video_root
        if root is None:
            return
        for i in range(root.childCount()):
            it = root.child(i)
            if it.data(0, Qt.UserRole) == (kind, rid):
                self._building = True
                self.tree.setCurrentItem(it)
                self._building = False
                return

    def _delete(self) -> None:
        kind, rid = self.current()
        if kind:
            self.deleteRequested.emit(kind, rid)

    def _track(self) -> None:
        kind, rid = self.current()
        if kind == "video":
            self.trackRequested.emit(rid)


class PIIDialog(QDialog):
    """Review automatically suggested PII and choose which to redact."""

    def __init__(self, suggestions: list[Suggestion], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Suggested sensitive speech")
        self.resize(640, 420)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Rule-based suggestions from the transcript. Check the items to redact. "
                             "Always review the transcript yourself as well."))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Redact", "Type", "Time", "Text", "Confidence"])
        self.tree.header().setSectionResizeMode(3, QHeaderView.Stretch)
        for s in suggestions:
            it = QTreeWidgetItem(["", s.kind, _fmt(s.start), s.text, f"{s.score:.0%}"])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(0, Qt.Checked if s.score >= 0.6 else Qt.Unchecked)
            it.setData(0, Qt.UserRole, s)
            self.tree.addTopLevelItem(it)
        lay.addWidget(self.tree, 1)
        row = QHBoxLayout()
        b_all = QPushButton("Check all")
        b_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        b_none = QPushButton("Uncheck all")
        b_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        row.addWidget(b_all)
        row.addWidget(b_none)
        row.addStretch(1)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Redact checked")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _set_all(self, state) -> None:
        for i in range(self.tree.topLevelItemCount()):
            self.tree.topLevelItem(i).setCheckState(0, state)

    def chosen(self) -> list[Suggestion]:
        out = []
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if it.checkState(0) == Qt.Checked:
                out.append(it.data(0, Qt.UserRole))
        return out
