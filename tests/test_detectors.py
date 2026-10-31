import numpy as np

from videoredact.core.model import BBox, Span, VideoTrack
from videoredact.vision.detectors.base import Detection, Detector, nms_merge
from videoredact.vision.detectors.wrappers import EdgeAssist, EnsembleDetector, HeadFallback, MultiScaleDetector, TiledDetector


class Recorder(Detector):
    """Returns the boxes given per call (normalized to whatever frame it is handed) and
    records the frame sizes it saw."""
    labels = {"face"}

    def __init__(self, per_call):
        self.per_call, self.sizes, self.name = list(per_call), [], "rec"

    def detect(self, frames):
        out = []
        for f in frames:
            self.sizes.append(f.shape[:2])
            dets = self.per_call.pop(0) if self.per_call else []
            out.append([Detection(b, "face", s, self.name, landmarks=lm) for b, s, lm in dets])
        return out


def test_edge_assist_adds_border_faces_and_drops_mirror_phantoms():
    W, H = 640, 360
    ea = EdgeAssist(Recorder([]), pad_frac=0.1, min_pad=48)
    pad = ea.pad_for(W, H)
    assert pad == 64
    PW, PH = W + 2 * pad, H + 2 * pad
    # Padded pass, in padded coordinates: (a) a face straddling the real left edge, half outside;
    # (b) a face entirely inside the pad (mirror image of an interior face); (c) an interior face
    # the normal pass also found (same box, different score).
    a = BBox.from_pixels(pad - 40, pad + 100, pad + 40, pad + 180, PW, PH)
    b = BBox.from_pixels(4, pad + 100, 60, pad + 180, PW, PH)
    c = BBox.from_pixels(pad + 300, pad + 100, pad + 380, pad + 180, PW, PH)
    lm = np.array([[(pad - 20) / PW, (pad + 120) / PH]] * 5, dtype=np.float32)
    normal = Recorder([[(BBox.from_pixels(300, 100, 380, 180, W, H), 0.7, None)]])
    padded = Recorder([[(a, 0.9, lm), (b, 0.8, None), (c, 0.95, None)]])
    ea.inner, ea.pad_inner = normal, padded
    dets = ea.detect([np.zeros((H, W, 3), np.uint8)])[0]
    assert normal.sizes == [(H, W)] and padded.sizes == [(PH, PW)]
    # the normal pass's box comes first and unchanged; the border face is added; the phantom
    # (b) and the duplicate (c) are not
    assert [(round(d.score, 2), d.source) for d in dets] == [(0.7, "rec"), (0.9, "edge-assist")]
    interior, edge = dets
    assert interior.bbox.to_pixels(W, H) == (300, 100, 380, 180)
    assert edge.bbox.to_pixels(W, H) == (0, 100, 40, 180)    # clipped to the real frame
    # landmarks are mapped back too (and may lie outside the frame)
    assert abs(edge.landmarks[0, 0] * W - (-20)) < 1e-3 and abs(edge.landmarks[0, 1] * H - 120) < 1e-3


def test_edge_assist_never_changes_the_normal_pass():
    """Union-only: with nothing new in the pad, the output is exactly the inner detector's."""
    W, H = 400, 300
    box = BBox.from_pixels(100, 50, 160, 120, W, H)
    ea = EdgeAssist(Recorder([[(box, 0.95, None)]]), pad_inner=Recorder([]))
    dets = ea.detect([np.zeros((H, W, 3), np.uint8)])[0]
    assert len(dets) == 1 and dets[0].bbox.to_pixels(W, H) == (100, 50, 160, 120) and dets[0].source == "rec"


class Fake(Detector):
    """Returns fixed normalized boxes regardless of input (so wrappers can be tested)."""
    labels = {"face", "person"}

    def __init__(self, dets, name="fake"):
        self._dets, self.name, self.calls = dets, name, 0

    def detect(self, frames):
        self.calls += len(frames)
        return [[Detection(b, l, s, self.name) for b, l, s in self._dets] for _ in frames]


def test_nms_merge_fuses_overlaps_and_keeps_classes():
    a = Detection(BBox(0.1, 0.1, 0.2, 0.2), "face", 0.9, "a")
    b = Detection(BBox(0.11, 0.11, 0.2, 0.2), "face", 0.7, "b")
    c = Detection(BBox(0.6, 0.6, 0.2, 0.2), "face", 0.8, "a")
    p = Detection(BBox(0.1, 0.1, 0.2, 0.2), "person", 0.8, "a")
    out = nms_merge([a, b, c, p], 0.5)
    assert len(out) == 3
    faces = [d for d in out if d.label == "face"]
    assert len(faces) == 2 and any("a+b" == d.source for d in faces)


def test_tiled_maps_tile_boxes_back_to_frame():
    inner = Fake([(BBox(0.0, 0.0, 0.5, 0.5), "face", 0.9)])
    tiled = TiledDetector(inner, tile=640, overlap=0.0, min_side=1000, full_frame=False)
    frame = np.zeros((1280, 1280, 3), np.uint8)
    dets = tiled.detect([frame])[0]
    # 2x2 non-overlapping tiles -> inner's box (top-left half of each tile) maps to 4 distinct frame boxes
    assert len(dets) == 4
    assert sorted((round(d.bbox.x, 2), round(d.bbox.y, 2)) for d in dets) == [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)]
    assert all(abs(d.bbox.w - 0.25) < 1e-6 for d in dets)
    # on a 720 px tall frame the two vertically overlapping tiles see the same object and are fused
    assert len(tiled.detect([np.zeros((720, 1280, 3), np.uint8)])[0]) == 2


def test_tiled_skips_small_frames():
    inner = Fake([(BBox(0.1, 0.1, 0.1, 0.1), "face", 0.9)])
    tiled = TiledDetector(inner, tile=640, min_side=1000)
    tiled.detect([np.zeros((360, 640, 3), np.uint8)])
    assert inner.calls == 1  # full frame only


def test_multiscale_and_ensemble_dedupe():
    inner = Fake([(BBox(0.2, 0.2, 0.1, 0.1), "face", 0.9)])
    ms = MultiScaleDetector(inner, [1.0, 1.5])
    assert len(ms.detect([np.zeros((200, 200, 3), np.uint8)])[0]) == 1
    ens = EnsembleDetector([Fake([(BBox(0.2, 0.2, 0.1, 0.1), "face", 0.9)], "x"),
                            Fake([(BBox(0.21, 0.2, 0.1, 0.1), "face", 0.6)], "y")])
    out = ens.detect([np.zeros((200, 200, 3), np.uint8)])[0]
    assert len(out) == 1 and out[0].source == "x+y"


def test_head_fallback_adds_head_only_when_face_missing_and_pads_faces():
    person = (BBox(0.4, 0.1, 0.2, 0.8), "person", 0.9)
    face_in_head = (BBox(0.47, 0.12, 0.06, 0.08), "face", 0.8)
    hf = HeadFallback(Fake([person]), face_pad=0.25, as_face=False)
    out = hf.detect([np.zeros((10, 10, 3), np.uint8)])[0]
    assert [d.label for d in out] == ["person", "head"]
    hf_face = HeadFallback(Fake([person]))
    out_f = hf_face.detect([np.zeros((10, 10, 3), np.uint8)])[0]
    assert [(d.label, d.source) for d in out_f][1] == ("face", "head-fallback")
    head = out[1].bbox
    assert abs(head.y - 0.1) < 1e-9 and abs(head.h - 0.8 * 0.22) < 1e-9
    hf2 = HeadFallback(Fake([person, face_in_head]), face_pad=0.25)
    out2 = hf2.detect([np.zeros((10, 10, 3), np.uint8)])[0]
    assert [d.label for d in out2] == ["person", "face"]
    padded = next(d for d in out2 if d.label == "face").bbox
    assert abs(padded.w - 0.06 * 1.5) < 1e-9 and abs(padded.h - 0.08 * 1.5) < 1e-9
    # wide (lying) person boxes get no head heuristic
    hf3 = HeadFallback(Fake([(BBox(0.1, 0.5, 0.8, 0.3), "person", 0.9)]), as_face=False)
    assert [d.label for d in hf3.detect([np.zeros((10, 10, 3), np.uint8)])[0]] == ["person"]


def test_track_trim_extend_flags():
    t = VideoTrack(label="face", source="auto", note="auto-detected, 1 hits, avg conf 0.45 (single detection - review)",
                   spans=[Span(100, 200, {100: BBox(.1, .1, .1, .1), 200: BBox(.5, .5, .1, .1)})])
    assert set(t.review_flags(30.0)) >= {"single", "low-conf"}
    t.trim_start(150)
    assert t.first_frame() == 150 and t.bbox_at(150) is not None and abs(t.bbox_at(150).x - 0.3) < 1e-6
    t.trim_end(180)
    assert t.last_frame() == 180 and t.bbox_at(190) is None
    t.extend_to(220)
    assert t.last_frame() == 220 and t.bbox_at(220) is not None
    t.trim_end(100)      # cut before the span start -> span removed
    assert not t.spans
