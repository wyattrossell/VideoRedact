"""Suggest personally-identifiable information in a transcript.

Rule-based (no model download): phone numbers, SSNs, dates of birth,
street addresses, license plates, spelled-out digit runs, e-mail addresses,
and words following name cues ("my name is", "this is officer"). Each
suggestion maps back to the Word objects so it can be turned into an
AudioRedaction with one click. Microsoft Presidio (MIT) can be plugged in
later for NER-based names; see docs/RESEARCH.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from videoredact.core.model import Segment, Word
from .transcribe import normalize_token

NUMBER_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
NAME_CUES = [
    ("my", "name", "is"), ("name", "is"), ("name", "as"), ("name", "was"), ("goes", "by"), ("known", "as"), ("this", "is", "officer"), ("this", "is", "detective"),
    ("i'm", "officer"), ("i", "am", "officer"), ("officer",), ("detective",), ("sergeant",),
    ("mr",), ("mrs",), ("ms",), ("miss",), ("dr",), ("named",), ("called",),
]
ADDRESS_SUFFIX = {"street", "st", "avenue", "ave", "road", "rd", "boulevard", "blvd", "drive", "dr",
                  "lane", "ln", "court", "ct", "way", "place", "pl", "circle", "highway", "hwy", "parkway"}
DOB_CUES = {("date", "of", "birth"), ("born",), ("birthday",), ("dob",)}
MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
          "sept", "oct", "nov", "dec"}


@dataclass
class Suggestion:
    kind: str          # phone, ssn, dob, address, plate, email, name, digits
    words: list[Word]
    text: str
    score: float

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end


def _digit_value(tok: str) -> str | None:
    if tok.isdigit():
        return tok
    return NUMBER_WORDS.get(tok)


def suggest_pii(segments: list[Segment]) -> list[Suggestion]:
    words = [w for s in segments for w in s.words]
    norm = [normalize_token(w.text) for w in words]
    n = len(words)
    out: list[Suggestion] = []
    used: set[int] = set()

    def add(kind: str, i0: int, i1: int, score: float) -> None:
        idx = range(i0, i1)
        if any(i in used for i in idx):
            return
        used.update(idx)
        ws = words[i0:i1]
        out.append(Suggestion(kind, ws, " ".join(w.text for w in ws), score))

    # e-mail (whisper usually writes it as one token or "name at domain dot com")
    for i, tok in enumerate(norm):
        if "@" in words[i].text and "." in words[i].text:
            add("email", i, i + 1, 0.9)
    for i in range(n - 4):
        if norm[i + 1] == "at" and norm[i + 3] == "dot" and norm[i + 4] in {"com", "org", "net", "gov", "edu"}:
            add("email", i, i + 5, 0.8)

    # Digit runs: collect sequences of digits / number words
    i = 0
    while i < n:
        j = i
        digits = ""
        while j < n:
            v = _digit_value(norm[j])
            if v is None:
                break
            digits += v
            j += 1
        if j > i:
            L = len(digits)
            raw = " ".join(words[k].text for k in range(i, j))
            if L >= 10 and L <= 11:
                add("phone", i, j, 0.9)
            elif L == 9 or re.search(r"\d{3}-\d{2}-\d{4}", raw):
                add("ssn", i, j, 0.9)
            elif L >= 7 and L <= 8:
                add("phone", i, j, 0.6)
            elif L >= 5:
                add("digits", i, j, 0.4)
            # street address: number followed by words then a street suffix
            elif 1 <= L <= 5 and j < n:
                for k in range(j + 1, min(n, j + 5)):
                    if norm[k] in ADDRESS_SUFFIX:
                        add("address", i, k + 1, 0.8)
                        break
            i = j
        else:
            i += 1

    # Dates of birth: cue followed within 8 tokens by a month or digits
    for i in range(n):
        for cue in DOB_CUES:
            L = len(cue)
            if tuple(norm[i:i + L]) == cue:
                for k in range(i + L, min(n, i + L + 8)):
                    if norm[k] in MONTHS or _digit_value(norm[k]) is not None:
                        end = k + 1
                        while end < n and end < k + 6 and (_digit_value(norm[end]) is not None or norm[end] in {"of", "the", "nineteen", "twenty", "hundred", "thousand"} or norm[end].endswith(("th", "st", "nd", "rd"))):
                            end += 1
                        add("dob", k, end, 0.75)
                        break

    # Names: capitalised word(s) following a cue
    for i in range(n):
        for cue in NAME_CUES:
            L = len(cue)
            if tuple(norm[i:i + L]) == cue and i + L < n:
                k = i + L
                end = k
                while end < n and end < k + 3 and words[end].text[:1].isupper() and norm[end] not in {"i"}:
                    end += 1
                if end > k:
                    add("name", k, end, 0.6 if L == 1 else 0.8)

    # License plates: mixed letters/digits runs like "ABC 1234" / "7 X R A 2 1 9"
    for i in range(n):
        if norm[i] in {"plate", "plates", "tag", "registration"} and i + 1 < n:
            k = i + 1
            while k < n and norm[k] in {"is", "number", "of", "reads", "was", "the"}:
                k += 1
            end = k
            while end < n and end < k + 8 and (len(norm[end]) <= 3 or _digit_value(norm[end]) is not None):
                end += 1
            if end - k >= 2:
                add("plate", k, end, 0.7)

    out.sort(key=lambda s: s.start)
    return out
