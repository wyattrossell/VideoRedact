"""Background task runner with a cancellable progress dialog."""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QWidget

# fn signature: fn(progress: Callable[[float, str], None], cancel: Callable[[], bool]) -> Any
TaskFn = Callable[[Callable[[float, str], None], Callable[[], bool]], Any]


class _Worker(QThread):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: TaskFn, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            result = self.fn(lambda p, m: self.progress.emit(float(p), str(m)), lambda: self._cancel)
            self.done.emit(result)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


def run_task(parent: QWidget, title: str, fn: TaskFn, on_done: Callable[[Any], None],
             on_fail: Optional[Callable[[str], None]] = None, cancellable: bool = True) -> _Worker:
    """Run fn in a thread while showing a modal progress dialog."""
    dlg = QProgressDialog(title, "Cancel" if cancellable else None, 0, 1000, parent)
    dlg.setWindowTitle("VideoRedact")
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setMinimumDuration(300)
    dlg.setMinimumWidth(420)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)
    worker = _Worker(fn, parent)

    def on_progress(p: float, msg: str) -> None:
        dlg.setValue(int(p * 1000))
        if msg:
            dlg.setLabelText(msg)

    def finish() -> None:
        dlg.reset()
        dlg.close()
        worker.deleteLater()

    def _done(result: Any) -> None:
        finish()
        if worker._cancel:
            return
        on_done(result)

    def _failed(msg: str) -> None:
        finish()
        if on_fail:
            on_fail(msg)
        else:
            box = QMessageBox(QMessageBox.Critical, "Task failed", msg.split("\n")[0], parent=parent)
            box.setDetailedText(msg)
            box.exec()

    worker.progress.connect(on_progress)
    worker.done.connect(_done)
    worker.failed.connect(_failed)
    if cancellable:
        dlg.canceled.connect(worker.cancel)
    worker.start()
    return worker
