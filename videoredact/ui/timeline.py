"""Timeline widget: time ruler, audio-redaction lane, one lane per video track,
playhead. Click/drag to seek, click a bar to select it, Ctrl+wheel to zoom.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QRectF, Qt, Signal, QPointF
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QFontMetrics
from PySide6.QtWidgets import QScrollArea, QWidget, QSizePolicy

from videoredact.core.model import Project

LANE_H = 18
RULER_H = 22
LABEL_W = 110
LABEL_COLORS = {
    "face": QColor(255, 140, 0), "screen": QColor(0, 170, 255), "document": QColor(170, 90, 255),
    "phone": QColor(0, 200, 140), "person": QColor(255, 80, 160), "license_plate": QColor(255, 220, 0),
}


class TimelineCanvas(QWidget):
    seekRequested = Signal(float)
    audioSelected = Signal(str)
    trackSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[Project] = None
        self.duration = 1.0
        self.playhead = 0.0
        self.px_per_s = 40.0
        self.selected_id: Optional[str] = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._dragging = False
        self.setFont(QFont("Segoe UI", 8))

    # ---- geometry --------------------------------------------------------
    def set_project(self, p: Optional[Project]) -> None:
        self.project = p
        self.duration = max(1.0, p.media.duration if p else 1.0)
        self._relayout()

    def n_lanes(self) -> int:
        return 1 + (len(self.project.video_tracks) if self.project else 0)

    def _relayout(self) -> None:
        w = int(LABEL_W + self.duration * self.px_per_s) + 20
        h = RULER_H + self.n_lanes() * LANE_H + 6
        self.setMinimumSize(w, h)
        self.setFixedHeight(h)
        self.resize(w, h)
        self.update()

    def set_zoom(self, px_per_s: float) -> None:
        self.px_per_s = max(2.0, min(2000.0, px_per_s))
        self._relayout()

    def x_of(self, t: float) -> float:
        return LABEL_W + t * self.px_per_s

    def t_of(self, x: float) -> float:
        return min(self.duration, max(0.0, (x - LABEL_W) / self.px_per_s))

    def set_playhead(self, t: float) -> None:
        self.playhead = t
        self.update()

    # ---- painting --------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(40, 42, 46))
        fm = QFontMetrics(self.font())
        W = self.width()
        # ruler
        p.fillRect(QRectF(0, 0, W, RULER_H), QColor(55, 58, 63))
        step = _nice_step(self.px_per_s)
        t = 0.0
        p.setPen(QPen(QColor(200, 200, 200)))
        while t <= self.duration + 1e-6:
            x = self.x_of(t)
            p.drawLine(QPointF(x, RULER_H - 6), QPointF(x, RULER_H))
            p.drawText(QPointF(x + 2, RULER_H - 8), _fmt(t))
            t += step
        # lanes
        y = RULER_H
        lanes = [("Audio", None)] + ([(tr.label, tr) for tr in self.project.video_tracks] if self.project else [])
        for i, (name, tr) in enumerate(lanes):
            p.fillRect(QRectF(0, y, W, LANE_H), QColor(48, 50, 55) if i % 2 else QColor(44, 46, 50))
            p.setPen(QPen(QColor(210, 210, 210)))
            label = name if tr is None else f"{i}. {name}"
            p.drawText(QRectF(4, y, LABEL_W - 8, LANE_H), Qt.AlignVCenter | Qt.AlignLeft, fm.elidedText(label, Qt.ElideRight, LABEL_W - 10))
            if self.project:
                if tr is None:
                    for r in self.project.audio_redactions:
                        c = QColor(230, 70, 70) if r.enabled else QColor(120, 80, 80)
                        self._bar(p, r.start, r.end, y, c, r.id == self.selected_id)
                else:
                    fps = self.project.media.fps or 30.0
                    base = LABEL_COLORS.get(tr.label, QColor(255, 60, 60))
                    c = base if tr.enabled else QColor(base.red() // 2, base.green() // 2, base.blue() // 2)
                    for s in tr.spans:
                        self._bar(p, s.start_frame / fps, (s.end_frame + 1) / fps, y, c, tr.id == self.selected_id)
            y += LANE_H
        # playhead
        x = self.x_of(self.playhead)
        p.setPen(QPen(QColor(255, 255, 255), 1.5))
        p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        p.end()

    def _bar(self, p: QPainter, t0: float, t1: float, y: float, color: QColor, selected: bool) -> None:
        x0, x1 = self.x_of(t0), self.x_of(t1)
        r = QRectF(x0, y + 3, max(2.0, x1 - x0), LANE_H - 6)
        p.setBrush(QBrush(color))
        p.setPen(QPen(Qt.white if selected else color.darker(150), 2 if selected else 1))
        p.drawRoundedRect(r, 3, 3)

    # ---- interaction -----------------------------------------------------
    def _hit(self, pos) -> Optional[tuple[str, str]]:
        if not self.project or pos.y() < RULER_H:
            return None
        lane = int((pos.y() - RULER_H) // LANE_H)
        t = self.t_of(pos.x())
        if lane == 0:
            for r in self.project.audio_redactions:
                if r.start <= t <= r.end:
                    return ("audio", r.id)
        elif 1 <= lane <= len(self.project.video_tracks):
            tr = self.project.video_tracks[lane - 1]
            fps = self.project.media.fps or 30.0
            f = int(t * fps)
            if tr.bbox_at(f) is not None:
                return ("video", tr.id)
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        hit = self._hit(event.position())
        if hit:
            self.selected_id = hit[1]
            (self.audioSelected if hit[0] == "audio" else self.trackSelected).emit(hit[1])
            self.update()
        if event.position().x() >= LABEL_W:
            self._dragging = True
            self.seekRequested.emit(self.t_of(event.position().x()))

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.seekRequested.emit(self.t_of(event.position().x()))

    def mouseReleaseEvent(self, event):
        self._dragging = False


class Timeline(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = TimelineCanvas()
        self.setWidget(self.canvas)
        self.setWidgetResizable(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(RULER_H + 3 * LANE_H + 30)
        self.setMaximumHeight(RULER_H + 10 * LANE_H + 30)
        self.setStyleSheet("QScrollArea { background: #28292d; border: 0; }")

    def set_project(self, p: Optional[Project]) -> None:
        self.canvas.set_project(p)
        if p and p.media.duration:
            # zoom to fit on load
            avail = max(200, self.viewport().width() - LABEL_W - 30)
            self.canvas.set_zoom(avail / p.media.duration)
        self.setMaximumHeight(min(RULER_H + max(3, self.canvas.n_lanes()) * LANE_H + 30, 400))

    def refresh(self) -> None:
        self.canvas._relayout()
        self.setMaximumHeight(min(RULER_H + max(3, self.canvas.n_lanes()) * LANE_H + 30, 400))

    def set_playhead(self, t: float, follow: bool = True) -> None:
        self.canvas.set_playhead(t)
        if follow:
            x = int(self.canvas.x_of(t))
            hb = self.horizontalScrollBar()
            vw = self.viewport().width()
            if x < hb.value() + LABEL_W or x > hb.value() + vw - 20:
                hb.setValue(max(0, x - vw // 3))

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            f = 1.25 if event.angleDelta().y() > 0 else 0.8
            # keep the time under the cursor fixed
            mx = event.position().x() + self.horizontalScrollBar().value()
            t = self.canvas.t_of(mx)
            self.canvas.set_zoom(self.canvas.px_per_s * f)
            self.horizontalScrollBar().setValue(int(self.canvas.x_of(t) - event.position().x()))
            event.accept()
        else:
            # horizontal scroll with plain wheel
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - event.angleDelta().y())
            event.accept()


def _nice_step(px_per_s: float) -> float:
    target_px = 90
    for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600):
        if s * px_per_s >= target_px:
            return s
    return 3600


def _fmt(t: float) -> str:
    if t < 60 and t != int(t):
        return f"{t:.1f}s"
    m, s = divmod(int(round(t)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
