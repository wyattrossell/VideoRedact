"""Non-blocking background jobs with a small status panel.

Long operations (tracking, detection, transcription, export) run in a
QThread while the window stays fully usable. Each job shows as a row with a
progress bar and a Cancel button in the Jobs dock. Jobs can stream partial
results to the UI through `partial(payload)` so, for example, a transcript
fills in while recognition is still running and a tracked box appears on
the timeline as the tracker advances.

Job function signature:
    fn(progress: Callable[[float, str], None],
       cancel: Callable[[], bool],
       partial: Callable[[object], None]) -> result
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (QDockWidget, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget, QMessageBox)

JobFn = Callable[[Callable[[float, str], None], Callable[[], bool], Callable[[Any], None]], Any]


class _JobThread(QThread):
    progress = Signal(float, str)
    partial = Signal(object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: JobFn):
        super().__init__()
        self.fn = fn
        self.cancelled = False

    def run(self) -> None:
        try:
            res = self.fn(lambda p, m: self.progress.emit(float(p), str(m)),
                          lambda: self.cancelled,
                          lambda payload: self.partial.emit(payload))
            self.done.emit(res)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class JobRow(QWidget):
    def __init__(self, title: str, thread: _JobThread):
        super().__init__()
        self.thread = thread
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)
        top = QHBoxLayout()
        self.title = QLabel(f"<b>{title}</b>")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFixedWidth(64)
        self.cancel_btn.clicked.connect(self._cancel)
        top.addWidget(self.title, 1)
        top.addWidget(self.cancel_btn)
        lay.addLayout(top)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(True)
        self.bar.setFixedHeight(14)
        lay.addWidget(self.bar)
        self.msg = QLabel("Starting…")
        self.msg.setStyleSheet("color: gray")
        lay.addWidget(self.msg)
        self.setStyleSheet("JobRow { border-bottom: 1px solid #555; }")

    def _cancel(self) -> None:
        self.thread.cancelled = True
        self.cancel_btn.setEnabled(False)
        self.msg.setText("Cancelling…")

    def set_progress(self, p: float, m: str) -> None:
        self.bar.setValue(int(p * 1000))
        if m:
            self.msg.setText(m)


class JobsPanel(QDockWidget):
    """Dock listing running jobs. Call run() to start one."""
    jobsChanged = Signal(int)  # number of running jobs

    def __init__(self, parent=None):
        super().__init__("Jobs", parent)
        self.setObjectName("JobsDock")
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self._rows: list[JobRow] = []
        self._threads: list[_JobThread] = []
        inner = QWidget()
        self._lay = QVBoxLayout(inner)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self._lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        self.setWidget(scroll)
        self.setMinimumHeight(90)
        self.hide()

    @property
    def running(self) -> int:
        return len(self._rows)

    def run(self, title: str, fn: JobFn, on_done: Callable[[Any], None],
            on_partial: Optional[Callable[[Any], None]] = None,
            on_fail: Optional[Callable[[str], None]] = None,
            on_cancel: Optional[Callable[[], None]] = None) -> _JobThread:
        th = _JobThread(fn)
        row = JobRow(title, th)
        self._lay.insertWidget(self._lay.count() - 1, row)
        self._rows.append(row)
        self._threads.append(th)
        self.show()
        self.jobsChanged.emit(self.running)

        def finish() -> None:
            self._lay.removeWidget(row)
            row.deleteLater()
            if row in self._rows:
                self._rows.remove(row)
            if th in self._threads:
                self._threads.remove(th)
            th.deleteLater()
            if not self._rows:
                self.hide()
            self.jobsChanged.emit(self.running)

        def _done(res: Any) -> None:
            cancelled = th.cancelled
            finish()
            if cancelled:
                if on_cancel:
                    on_cancel()
                return
            on_done(res)

        def _failed(msg: str) -> None:
            finish()
            if on_fail:
                on_fail(msg)
            else:
                box = QMessageBox(QMessageBox.Critical, f"{title} failed", msg.split("\n")[0], parent=self.parent())
                box.setDetailedText(msg)
                box.exec()

        th.progress.connect(row.set_progress)
        if on_partial:
            th.partial.connect(on_partial)
        th.done.connect(_done)
        th.failed.connect(_failed)
        th.start()
        return th

    def cancel_all(self) -> None:
        for th in self._threads:
            th.cancelled = True

    def wait_all(self, ms: int = 5000) -> None:
        for th in list(self._threads):
            th.wait(ms)
