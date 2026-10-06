"""Transcribe tests/data/sample.mp4 and print word timings (downloads whisper model on first run)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from videoredact.core.media import decode_audio
from videoredact.audio.transcribe import Transcriber, find_matches
from videoredact.audio.pii import suggest_pii

size = sys.argv[1] if len(sys.argv) > 1 else "small"
t0 = time.time()
tr = Transcriber(size)
print(f"model {size} loaded in {time.time()-t0:.1f}s")
audio, sr = decode_audio("tests/data/sample.mp4", sample_rate=16000, mono=True)
t0 = time.time()
segs = tr.transcribe(audio[:, 0], duration=len(audio)/sr, language="en", progress=lambda p, m: None)
print(f"transcribed {len(audio)/sr:.1f}s of audio in {time.time()-t0:.1f}s")
for s in segs:
    print(f"[{s.start:6.2f}-{s.end:6.2f}] {s.text}")
    print("    " + " ".join(f"{w.text}@{w.start:.2f}" for w in s.words))
print("matches 'Michael Carter':", [(h[0].start, h[-1].end) for h in find_matches(segs, "Michael Carter")])
print("PII:", [(x.kind, x.text) for x in suggest_pii(segs)])
