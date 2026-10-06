"""Application entry point."""
from __future__ import annotations

import os
import sys


def main() -> None:
    # Qt Multimedia backend. On Windows the Media Foundation backend ("windows") plays
    # reliably with audio; the bundled FFmpeg backend stalled with audio output on some
    # machines (see NOTES.md). Users can switch in Settings (needs restart).
    from videoredact.paths import settings
    default_backend = "windows" if sys.platform == "win32" else "ffmpeg"
    os.environ.setdefault("QT_MEDIA_BACKEND", settings.get("media_backend") or default_backend)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from videoredact.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("VideoRedact")
    app.setOrganizationName("VideoRedact")
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        win.open_path(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
