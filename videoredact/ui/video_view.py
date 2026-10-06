"""Video preview with redaction overlay and box drawing.

Frames arrive as numpy BGR images from the preview player (ui/player.py),
are optionally redacted exactly as the export would do it (so what you see
is what you get), and displayed on a QGraphicsPixmapItem.
Scene coordinates == source pixel coordinates, which makes box drawing and
normalisation trivial.

Interaction:
  * left-drag on empty area      -> draw a new box (boxDrawn signal)
  * click a box                  -> select it (trackSelected signal)
  * drag a selected box / handle -> move / resize (boxEdited signal)
  * Delete                       -> deleteRequested signal
  * wheel                        -> zoom, double-click -> fit
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
                               QGraphicsView)

from videoredact.core.model import BBox, VideoStyle, VideoTrack
from videoredact.core.video_redact import redact_frame

LABEL_COLORS = {
    "face": QColor(255, 140, 0), "screen": QColor(0, 170, 255), "document": QColor(170, 90, 255),
    "phone": QColor(0, 200, 140), "person": QColor(255, 80, 160), "license_plate": QColor(255, 220, 0),
}
DEFAULT_COLOR = QColor(255, 60, 60)
HANDLE = 7.0


class BoxItem(QGraphicsRectItem):
    def __init__(self, track: VideoTrack, rect: QRectF, view: "VideoView"):
        super().__init__(rect)
        self.track = track
        self.view = view
        self._drag: Optional[str] = None
        self._start_rect = QRectF()
        self._start_pos = QPointF()
        color = LABEL_COLORS.get(track.label, DEFAULT_COLOR)
        self.setPen(QPen(color, 2, Qt.SolidLine))
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    def paint(self, painter: QPainter, option, widget=None):
        pen = self.pen()
        if self.isSelected():
            pen.setWidthF(3)
            pen.setStyle(Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(self.brush())
        painter.drawRect(self.rect())
        painter.setPen(QPen(Qt.white))
        painter.setBrush(QBrush(QColor(0, 0, 0, 150)))
        r = self.rect()
        label = f"{self.track.label}"
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 6
        painter.drawRect(QRectF(r.left(), r.top() - fm.height() - 2, tw, fm.height() + 2))
        painter.drawText(QPointF(r.left() + 3, r.top() - 4), label)
        if self.isSelected():
            painter.setBrush(QBrush(Qt.white))
            painter.setPen(QPen(Qt.black, 1))
            s = HANDLE / max(0.2, self.view.transform().m11())
            for p in self._handles(s):
                painter.drawRect(QRectF(p.x() - s / 2, p.y() - s / 2, s, s))

    def _handles(self, s: float) -> list[QPointF]:
        r = self.rect()
        return [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]

    def _hit_handle(self, pos: QPointF) -> Optional[str]:
        s = HANDLE / max(0.2, self.view.transform().m11())
        names = ["tl", "tr", "bl", "br"]
        for n, p in zip(names, self._handles(s)):
            if abs(pos.x() - p.x()) <= s and abs(pos.y() - p.y()) <= s:
                return n
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        self.setSelected(True)
        self.view.trackSelected.emit(self.track.id)
        self._drag = self._hit_handle(event.pos()) or "move"
        self._start_rect = QRectF(self.rect())
        self._start_pos = event.scenePos()
        event.accept()

    def mouseMoveEvent(self, event):
        if not self._drag:
            return
        d = event.scenePos() - self._start_pos
        r = QRectF(self._start_rect)
        if self._drag == "move":
            r.translate(d)
        else:
            if "l" in self._drag:
                r.setLeft(r.left() + d.x())
            if "r" in self._drag:
                r.setRight(r.right() + d.x())
            if "t" in self._drag:
                r.setTop(r.top() + d.y())
            if "b" in self._drag:
                r.setBottom(r.bottom() + d.y())
            r = r.normalized()
        self.setRect(r)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._drag:
            self._drag = None
            if self.rect() != self._start_rect:
                self.view.boxEdited.emit(self.track.id, self.view.rect_to_bbox(self.rect()))
        event.accept()


class VideoView(QGraphicsView):
    boxDrawn = Signal(object)            # BBox
    trackSelected = Signal(str)          # track id
    boxEdited = Signal(str, object)      # track id, BBox
    deleteRequested = Signal()
    frameShown = Signal(int)             # frame index actually displayed
    backgroundClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QBrush(QColor(25, 25, 28)))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.pix_item = QGraphicsPixmapItem()
        self.pix_item.setZValue(0)
        self._scene.addItem(self.pix_item)
        self.width_px = 0
        self.height_px = 0
        self.fps = 30.0
        self.current_frame = 0
        self.tracks: list[VideoTrack] = []
        self.default_style = VideoStyle.BLACK
        self.preview_redactions = True
        self.draw_mode = True
        self.selected_track_id: Optional[str] = None
        self._rubber: Optional[QGraphicsRectItem] = None
        self._rubber_origin = QPointF()
        self._boxes: list[BoxItem] = []
        self._last_bgr: Optional[np.ndarray] = None
        self._fit_pending = True

    # ---- media -----------------------------------------------------------
    def set_media_props(self, width: int, height: int, fps: float) -> None:
        self.width_px, self.height_px, self.fps = width, height, fps or 30.0
        self._scene.setSceneRect(0, 0, max(1, width), max(1, height))
        self._fit_pending = True

    def show_still(self, bgr: np.ndarray, frame_idx: int) -> None:
        """Display a decoded BGR frame (from the player or a placeholder)."""
        h, w = bgr.shape[:2]
        if (w, h) != (self.width_px, self.height_px):
            self.set_media_props(w, h, self.fps)
        self._last_bgr = bgr
        self.current_frame = frame_idx
        self._render()
        self.frameShown.emit(frame_idx)

    def _render(self) -> None:
        if self._last_bgr is None:
            return
        bgr = self._last_bgr
        if self.preview_redactions and self.tracks:
            bgr = redact_frame(bgr, self.tracks, self.current_frame, self.default_style, in_place=False)
        h, w = bgr.shape[:2]
        qimg = QImage(bgr.data, w, h, w * 3, QImage.Format_BGR888)
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        if self._fit_pending:
            self.fit()
            self._fit_pending = False
        self.refresh_boxes()

    # ---- overlay ---------------------------------------------------------
    def set_tracks(self, tracks: list[VideoTrack], default_style: VideoStyle) -> None:
        self.tracks = tracks
        self.default_style = default_style
        self._render()

    def set_frame_index(self, idx: int) -> None:
        """Called when the playhead moves without a new video frame (audio-only / paused re-render)."""
        if idx != self.current_frame:
            self.current_frame = idx
            self._render()

    def refresh_boxes(self) -> None:
        for b in self._boxes:
            self._scene.removeItem(b)
        self._boxes = []
        if not self.width_px:
            return
        for t in self.tracks:
            bb = t.bbox_at(self.current_frame)
            if bb is None:
                continue
            rect = self.bbox_to_rect(bb)
            item = BoxItem(t, rect, self)
            self._scene.addItem(item)
            if t.id == self.selected_track_id:
                item.setSelected(True)
            self._boxes.append(item)

    def select_track(self, tid: Optional[str]) -> None:
        self.selected_track_id = tid
        for b in self._boxes:
            b.setSelected(b.track.id == tid)

    # ---- coordinate helpers --------------------------------------------
    def bbox_to_rect(self, bb: BBox) -> QRectF:
        return QRectF(bb.x * self.width_px, bb.y * self.height_px, bb.w * self.width_px, bb.h * self.height_px)

    def rect_to_bbox(self, r: QRectF) -> BBox:
        return BBox.from_pixels(r.left(), r.top(), r.right(), r.bottom(), self.width_px, self.height_px)

    # ---- view ------------------------------------------------------------
    def fit(self) -> None:
        if self.width_px:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def wheelEvent(self, event):
        f = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)

    def mouseDoubleClickEvent(self, event):
        self.fit()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.draw_mode and self.width_px:
            item = self.itemAt(event.pos())
            if isinstance(item, BoxItem):
                return super().mousePressEvent(event)
            self.select_track(None)
            self.backgroundClicked.emit()
            self._rubber_origin = self.mapToScene(event.pos())
            self._rubber = QGraphicsRectItem(QRectF(self._rubber_origin, self._rubber_origin))
            self._rubber.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            self._rubber.setZValue(20)
            self._scene.addItem(self._rubber)
            event.accept()
            return
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._rubber is not None:
            p = self.mapToScene(event.pos())
            self._rubber.setRect(QRectF(self._rubber_origin, p).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._rubber is not None:
            rect = self._rubber.rect().intersected(self._scene.sceneRect())
            self._scene.removeItem(self._rubber)
            self._rubber = None
            if rect.width() >= 4 and rect.height() >= 4:
                self.boxDrawn.emit(self.rect_to_bbox(rect))
            event.accept()
            return
        self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.selected_track_id:
            self.deleteRequested.emit()
            return
        super().keyPressEvent(event)
