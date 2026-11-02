import numpy as np
import pytest

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


# ------------------------------------------------------------------ licence plates ----
def test_plate_rows_are_filtered_like_privacycam_and_mapped_back():
    """YOLOv9-t rows in letterboxed 384 px input pixels -> frame boxes; tall boxes, huge boxes,
    low scores and duplicates are dropped; the output is capped at 20 per frame."""
    from videoredact.vision.detectors.plates import decode_plate_rows, letterbox, PLATE_MAX_PER_FRAME
    W, H = 1280, 720
    _, r, dx, dy = letterbox(np.zeros((H, W, 3), np.uint8), 384)
    assert abs(r - 0.3) < 1e-6 and dx == 0 and dy == 84
    def row(x1, y1, x2, y2, s):
        return [0, x1, y1, x2, y2, 0, s]
    rows = [row(100, 150, 160, 170, 0.9),      # a plate: 60x20 -> aspect 3
            row(100, 150, 110, 190, 0.9),      # taller than wide: dropped (aspect 0.25)
            row(0, 0, 383, 383, 0.95),         # fills the input: dropped (area)
            row(200, 200, 260, 220, 0.3),      # below threshold
            row(101, 151, 161, 171, 0.8)]      # duplicate of the first (IoU > 0.55)
    dets = decode_plate_rows(np.array(rows, np.float32), W, H, r, dx, dy, 0.5)
    assert len(dets) == 1 and dets[0].label == "license_plate" and abs(dets[0].score - 0.9) < 1e-6
    x0, y0, x1, y1 = dets[0].bbox.to_pixels(W, H)
    # 100..160 / 0.3 -> 333..533 wide; 150..170 - 84 offset / 0.3 -> 220..287 tall, padded 10 % each side
    assert (x0, x1) == (333, 533) and y0 == 213 and y1 == 293
    many = np.array([row(10 + 20 * i, 100, 10 + 20 * i + 12, 104, 0.6) for i in range(18)]
                    + [row(10 + 20 * i, 200, 10 + 20 * i + 12, 204, 0.6) for i in range(18)], np.float32)
    assert len(decode_plate_rows(many, W, H, r, dx, dy, 0.5)) == PLATE_MAX_PER_FRAME


def test_vehicle_gate_keeps_plates_on_vehicles_only():
    from videoredact.vision.detectors.plates import UNGATED_SOURCE, VehicleGate, VehiclePlates
    car = (BBox(0.5, 0.4, 0.4, 0.5), "vehicle", 0.8)
    on_car = (BBox(0.6, 0.7, 0.1, 0.04), "license_plate", 0.9)
    laptop = (BBox(0.1, 0.1, 0.2, 0.08), "license_plate", 0.95)    # the in-car screen, nowhere near a vehicle
    face = (BBox(0.1, 0.5, 0.1, 0.15), "face", 0.9)
    inner = Fake([car, on_car, laptop, face])
    inner.labels = {"vehicle", "license_plate", "face"}
    frame = [np.zeros((10, 10, 3), np.uint8)]
    out = VehicleGate(inner).detect(frame)[0]
    assert [(d.label, d.score) for d in out] == [("license_plate", 0.9), ("face", 0.9)]
    assert "vehicle" not in VehicleGate(inner).labels
    kept = VehicleGate(inner, keep_vehicles=True).detect(frame)[0]
    assert [d.label for d in kept] == ["vehicle", "license_plate", "face"]
    # the full wrapper passes the rejected plate on, marked for continuation only
    out = VehiclePlates(inner, None).detect(frame)[0]
    assert [(d.label, d.source) for d in out] == [("license_plate", "fake"), ("license_plate", UNGATED_SOURCE), ("face", "fake")]


def test_vehicle_crops_find_the_small_plate_and_map_it_back():
    """The full-frame pass misses a distant plate; the crop of the vehicle box finds it, and
    its box comes back in frame coordinates, inside the vehicle."""
    from videoredact.vision.detectors.plates import VehiclePlates
    from videoredact.vision.detectors.base import PerFrameDetector

    class CropPlate(PerFrameDetector):
        labels = {"license_plate"}
        name = "plate_fake"
        seen = []

        def _detect_frame(self, frame):
            self.seen.append(frame.shape[:2])
            # a plate in the lower middle of whatever crop it is handed
            return [Detection(BBox(0.4, 0.8, 0.2, 0.08), "license_plate", 0.7, self.name)]

    frame = np.zeros((360, 640, 3), np.uint8)
    car = (BBox(0.5, 0.25, 0.25, 0.5), "vehicle", 0.3)           # 160 x 180 px, taken at the low vehicle threshold
    inner = Fake([car])
    inner.labels = {"vehicle", "license_plate"}
    pd = CropPlate()
    out = VehiclePlates(inner, pd).detect([frame])[0]
    assert pd.seen and pd.seen[0][0] < 360 and pd.seen[0][1] < 640      # the detector saw a crop, not the frame
    assert len(out) == 1 and out[0].label == "license_plate" and out[0].source.endswith("+crop")
    b = out[0].bbox
    gx0, gy0 = 0.5 - 0.25 * 0.08, 0.25 - 0.5 * 0.08                       # the grown vehicle box
    gw, gh = 0.25 * 1.16, 0.5 * 1.16
    assert abs(b.x - (gx0 + 0.4 * gw)) < 0.01 and abs(b.y - (gy0 + 0.8 * gh)) < 0.01
    assert abs(b.w - 0.2 * gw) < 0.01 and abs(b.h - 0.08 * gh) < 0.01
    # without the crop pass the plate is never seen
    assert VehiclePlates(inner, pd, crops=False).detect([frame])[0] == []


def test_ungated_plates_only_continue_confirmed_plate_tracks():
    from videoredact.vision.auto_detect import _Live, continue_ungated_plates
    from videoredact.vision.detectors.plates import UNGATED_SOURCE
    confirmed = _Live("license_plate", 10, BBox(0.5, 0.5, 0.1, 0.05), 0.8)
    confirmed.hits = 3
    fresh = _Live("license_plate", 10, BBox(0.1, 0.1, 0.1, 0.05), 0.8)       # one hit only
    same = Detection(BBox(0.51, 0.5, 0.1, 0.05), "license_plate", 0.6, UNGATED_SOURCE)
    near_fresh = Detection(BBox(0.11, 0.1, 0.1, 0.05), "license_plate", 0.6, UNGATED_SOURCE)
    elsewhere = Detection(BBox(0.8, 0.8, 0.1, 0.05), "license_plate", 0.9, UNGATED_SOURCE)
    gated = Detection(BBox(0.3, 0.3, 0.1, 0.05), "license_plate", 0.6, "plate_yolov9t")
    out = continue_ungated_plates([same, near_fresh, elsewhere, gated], [confirmed, fresh])
    assert [d.bbox for d in out] == [same.bbox, gated.bbox]
    assert out[0].source != UNGATED_SOURCE and "continued" in out[0].source


def test_build_with_plates_adds_the_vehicle_gate():
    from videoredact.vision import models as reg
    if not reg.is_available("plate_yolov9t") or not reg.is_available("yolox_tiny"):
        pytest.skip("plate / object models not installed")
    from videoredact.vision.detectors import build, VehiclePlates, EnsembleDetector
    det = build({"face": "", "objects": ["license_plate"], "face_pad": 0})
    assert isinstance(det, VehiclePlates) and isinstance(det.inner, EnsembleDetector)
    names = sorted(m.name for m in det.inner.members)
    assert names == ["plate_yolov9t", "yolox_tiny"]
    yolox = next(m for m in det.inner.members if m.name == "yolox_tiny")
    assert yolox.labels == {"vehicle"} and yolox.label_conf == {"vehicle": 0.35}
    assert det.plate_det is det.inner.members[names.index("plate_yolov9t") if False else [m.name for m in det.inner.members].index("plate_yolov9t")]
    assert det.crops and det.keep_ungated
    assert "vehicle" not in det.labels and "license_plate" in det.labels
    # the gate can be switched off (bench comparisons, dialog option)
    assert not isinstance(build({"face": "", "objects": ["license_plate"], "plate_vehicle_gate": False, "face_pad": 0}), VehiclePlates)
