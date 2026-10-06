"""Render the application icon (installer/videoredact.ico + .png) with Qt.
A dark shield with a redaction bar: simple, readable at 16 px."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen, QBrush, QFont, QLinearGradient

OUT = Path(__file__).resolve().parents[1] / "installer"


def render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    s = size
    # shield
    path = QPainterPath()
    path.moveTo(s * 0.5, s * 0.04)
    path.lineTo(s * 0.92, s * 0.18)
    path.lineTo(s * 0.88, s * 0.58)
    path.cubicTo(s * 0.85, s * 0.80, s * 0.65, s * 0.92, s * 0.5, s * 0.97)
    path.cubicTo(s * 0.35, s * 0.92, s * 0.15, s * 0.80, s * 0.12, s * 0.58)
    path.lineTo(s * 0.08, s * 0.18)
    path.closeSubpath()
    g = QLinearGradient(0, 0, 0, s)
    g.setColorAt(0, QColor(40, 90, 160))
    g.setColorAt(1, QColor(18, 40, 80))
    p.fillPath(path, QBrush(g))
    p.setPen(QPen(QColor(200, 215, 240), max(1.0, s * 0.03)))
    p.drawPath(path)
    # "video frame" rectangle
    fr = QRectF(s * 0.26, s * 0.30, s * 0.48, s * 0.36)
    p.setPen(QPen(QColor(235, 240, 250), max(1.0, s * 0.035)))
    p.setBrush(QBrush(QColor(255, 255, 255, 30)))
    p.drawRoundedRect(fr, s * 0.04, s * 0.04)
    # redaction bar
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(QColor(10, 10, 12)))
    p.drawRect(QRectF(s * 0.33, s * 0.43, s * 0.34, s * 0.10))
    p.end()
    return img


def main():
    app = QGuiApplication(sys.argv)
    OUT.mkdir(exist_ok=True)
    base = render(256)
    base.save(str(OUT / "videoredact.png"))
    # ICO writer in Qt stores one image; Windows scales it. Use 256 px.
    ok = base.save(str(OUT / "videoredact.ico"), "ICO")
    print("icon written:", ok, OUT / "videoredact.ico")


if __name__ == "__main__":
    main()
