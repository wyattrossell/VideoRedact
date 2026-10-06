"""Word-level transcript view.

Each word is rendered as its own text fragment so selections map back to
Word objects. Redacted words are shown struck through on a red background;
the word under the playhead is highlighted. Right-click gives redaction
actions; a search box finds every occurrence of a word/phrase.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor, QFont, QAction, QTextDocument
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QTextEdit, QVBoxLayout,
                               QWidget, QCheckBox)

from videoredact.audio.transcribe import find_matches
from videoredact.core.model import Project, Segment, Word


class TranscriptView(QWidget):
    seekRequested = Signal(float)                 # seconds
    redactWords = Signal(list, str)               # [Word], source
    redactAllOf = Signal(str)                     # phrase
    unredactWords = Signal(list)                  # [Word]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Optional[Project] = None
        self._words: list[Word] = []
        self._ranges: list[tuple[int, int]] = []   # char ranges per word
        self._current_word = -1

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        top = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find word or phrase…  (Enter = next match)")
        self.search.returnPressed.connect(self.find_next)
        self.search.textChanged.connect(self._update_count)
        self.count_lbl = QLabel("")
        self.btn_all = QPushButton("Redact all matches")
        self.btn_all.clicked.connect(lambda: self.redactAllOf.emit(self.search.text().strip()))
        self.btn_all.setEnabled(False)
        top.addWidget(self.search, 1)
        top.addWidget(self.count_lbl)
        top.addWidget(self.btn_all)
        lay.addLayout(top)

        self.follow = QCheckBox("Follow playback")
        self.follow.setChecked(True)
        lay.addWidget(self.follow)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setContextMenuPolicy(Qt.CustomContextMenu)
        self.text.customContextMenuRequested.connect(self._menu)
        self.text.mouseDoubleClickEvent = self._double_click  # type: ignore[assignment]
        f = QFont()
        f.setPointSize(11)
        self.text.setFont(f)
        lay.addWidget(self.text, 1)

        self.hint = QLabel("Select words, then right-click to redact. Double-click a word to jump there.")
        self.hint.setStyleSheet("color: gray")
        lay.addWidget(self.hint)
        self._match_idx = -1

    # ---- building --------------------------------------------------------
    def set_project(self, project: Optional[Project]) -> None:
        self.project = project
        self.rebuild()

    def rebuild(self) -> None:
        self.text.clear()
        self._words, self._ranges = [], []
        if not self.project or not self.project.transcript:
            self.text.setPlaceholderText("No transcript yet. Use Audio ▸ Transcribe.")
            self._update_count()
            return
        cur = self.text.textCursor()
        cur.beginEditBlock()
        for seg in self.project.transcript:
            tfmt = QTextCharFormat()
            tfmt.setForeground(QColor(120, 120, 120))
            tfmt.setFontPointSize(8.5)
            cur.insertText(f"[{_fmt(seg.start)}]  ", tfmt)
            for w in seg.words:
                start = cur.position()
                cur.insertText(w.text, self._fmt_for(w))
                end = cur.position()
                self._words.append(w)
                self._ranges.append((start, end))
                cur.insertText(" ", QTextCharFormat())
            cur.insertBlock()
        cur.endEditBlock()
        self.text.moveCursor(QTextCursor.Start)
        self._update_count()

    def _fmt_for(self, w: Word) -> QTextCharFormat:
        f = QTextCharFormat()
        if self.project and self.project.word_is_redacted(w):
            f.setBackground(QColor(255, 90, 90))
            f.setForeground(QColor(255, 255, 255))
            f.setFontStrikeOut(True)
        elif w.confidence < 0.5:
            f.setForeground(QColor(150, 110, 40))  # low-confidence word
        return f

    def refresh_marks(self) -> None:
        """Re-apply redaction formatting without rebuilding."""
        if not self._words:
            return
        cur = self.text.textCursor()
        cur.beginEditBlock()
        for w, (s, e) in zip(self._words, self._ranges):
            cur.setPosition(s)
            cur.setPosition(e, QTextCursor.KeepAnchor)
            cur.setCharFormat(self._fmt_for(w))
        cur.endEditBlock()

    # ---- selection -------------------------------------------------------
    def selected_words(self) -> list[Word]:
        c = self.text.textCursor()
        if not c.hasSelection():
            i = self._word_at(c.position())
            return [self._words[i]] if i >= 0 else []
        a, b = sorted((c.selectionStart(), c.selectionEnd()))
        return [w for w, (s, e) in zip(self._words, self._ranges) if e > a and s < b]

    def _word_at(self, pos: int) -> int:
        for i, (s, e) in enumerate(self._ranges):
            if s <= pos <= e:
                return i
        return -1

    def _double_click(self, event) -> None:
        pos = self.text.cursorForPosition(event.pos()).position()
        i = self._word_at(pos)
        if i >= 0:
            self.seekRequested.emit(self._words[i].start)

    def _menu(self, pos) -> None:
        words = self.selected_words()
        menu = QMenu(self)
        if words:
            phrase = " ".join(w.text for w in words)
            short = phrase if len(phrase) < 40 else phrase[:37] + "…"
            a1 = QAction(f"Redact “{short}”", self)
            a1.triggered.connect(lambda: self.redactWords.emit(words, "manual"))
            menu.addAction(a1)
            n = len(find_matches(self.project.transcript, phrase)) if self.project else 0
            a2 = QAction(f"Redact every occurrence of “{short}”  ({n})", self)
            a2.triggered.connect(lambda: self.redactAllOf.emit(phrase))
            menu.addAction(a2)
            if self.project and any(self.project.word_is_redacted(w) for w in words):
                a3 = QAction("Remove redaction from selection", self)
                a3.triggered.connect(lambda: self.unredactWords.emit(words))
                menu.addAction(a3)
            menu.addSeparator()
            a4 = QAction("Jump to selection", self)
            a4.triggered.connect(lambda: self.seekRequested.emit(words[0].start))
            menu.addAction(a4)
        else:
            menu.addAction("Select words first").setEnabled(False)
        menu.exec(self.text.mapToGlobal(pos))

    # ---- playback highlight --------------------------------------------
    def set_time(self, t: float) -> None:
        idx = -1
        for i, w in enumerate(self._words):
            if w.start <= t < w.end + 0.05:
                idx = i
                break
            if w.start > t:
                break
        if idx == self._current_word:
            return
        self._current_word = idx
        sels = []
        if idx >= 0:
            s, e = self._ranges[idx]
            sel = QTextEdit.ExtraSelection()
            sel.cursor = self.text.textCursor()
            sel.cursor.setPosition(s)
            sel.cursor.setPosition(e, QTextCursor.KeepAnchor)
            sel.format.setBackground(QColor(255, 235, 120))
            sel.format.setFontWeight(QFont.Bold)
            sels.append(sel)
            if self.follow.isChecked():
                c = self.text.textCursor()
                c.setPosition(s)
                self.text.setTextCursor(c)
                self.text.ensureCursorVisible()
                c.clearSelection()
        self.text.setExtraSelections(sels)

    # ---- search ----------------------------------------------------------
    def _update_count(self) -> None:
        q = self.search.text().strip()
        if not q or not self.project:
            self.count_lbl.setText("")
            self.btn_all.setEnabled(False)
            self._match_idx = -1
            return
        n = len(find_matches(self.project.transcript, q))
        self.count_lbl.setText(f"{n} match{'es' if n != 1 else ''}")
        self.btn_all.setEnabled(n > 0)
        self._match_idx = -1

    def find_next(self) -> None:
        q = self.search.text().strip()
        if not q or not self.project:
            return
        hits = find_matches(self.project.transcript, q)
        if not hits:
            return
        self._match_idx = (self._match_idx + 1) % len(hits)
        words = hits[self._match_idx]
        i0 = self._words.index(words[0])
        i1 = self._words.index(words[-1])
        c = self.text.textCursor()
        c.setPosition(self._ranges[i0][0])
        c.setPosition(self._ranges[i1][1], QTextCursor.KeepAnchor)
        self.text.setTextCursor(c)
        self.text.ensureCursorVisible()
        self.seekRequested.emit(words[0].start)


def _fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
