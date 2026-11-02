"""Variable frame rate (phone footage): detected when a file is opened, normalised to a
constant-rate working copy whose audio is copied unchanged; original path and hash recorded."""
import json
import subprocess
from pathlib import Path

import pytest

from videoredact.core import media
from videoredact.core.media import frame_timing, nearest_standard_rate, normalize_cfr, probe

SAMPLE = Path("tests/data/sample.mp4")


@pytest.fixture(scope="module")
def vfr_clip(tmp_path_factory) -> Path:
    """4 s test pattern with a 440 Hz tone whose frame intervals alternate 45, 45, 9 ms."""
    out = tmp_path_factory.mktemp("vfr") / "phone.mp4"
    cmd = [media.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=4",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
           # a 90 kHz timebase so a 12 ms offset survives (testsrc's own 1/30 s timebase rounds it to zero);
           # no commas in the expression: they split the filtergraph
           "-vf", "settb=1/90000,setpts=PTS+0.012*(N-3*floor(N/3))/TB", "-fps_mode", "vfr", "-video_track_timescale", "90000",
           "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(out)]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        pytest.skip("ffmpeg could not synthesise a VFR clip: " + res.stderr.decode("utf-8", "replace")[-300:])
    return out


def test_nearest_standard_rate():
    assert nearest_standard_rate(29.83) == 29.97
    assert nearest_standard_rate(30.2) == 30.0
    assert nearest_standard_rate(59.5) == 59.94
    assert nearest_standard_rate(27.5) == 27.5        # nothing standard within 8 %
    assert nearest_standard_rate(0) == 30.0


def test_body_cam_style_clip_is_constant_rate():
    t = frame_timing(SAMPLE)
    assert t["frames"] >= 10 and t["vfr"] is False and t["irregular_frac"] < 0.02
    info = probe(str(SAMPLE), timing=True)
    assert info.vfr is False and info.avg_fps > 0
    # the scan is opt-in: a plain probe does not pay for it
    assert probe(str(SAMPLE)).avg_fps == 0.0


def test_vfr_clip_is_detected_and_normalised(tmp_path, vfr_clip):
    t = frame_timing(vfr_clip)
    assert t["vfr"] is True and t["irregular_frac"] > 0.2, t
    assert probe(str(vfr_clip), timing=True).vfr is True
    src_tags = media.media_tags(vfr_clip)
    dst = tmp_path / "phone_CFR.mp4"
    seen = []
    res = normalize_cfr(vfr_clip, dst, 30.0, progress=lambda p, m: seen.append(p))
    assert dst.exists() and not dst.with_name("phone_CFR.part.mp4").exists()
    assert res["audio_copied"] is True and res["fps"] == 30.0 and len(res["sha256"]) == 64
    assert seen and max(seen) <= 0.99
    after = frame_timing(dst)
    assert after["vfr"] is False, after
    info = probe(str(dst), timing=True)
    assert info.has_audio and info.audio_codec == "aac" and abs(info.duration - 4.0) < 0.3
    assert abs(info.frame_count - 120) <= 3 and abs(info.fps - 30.0) < 0.01
    # the copy carries none of the original's identifying tags either (same raw-pipe rule as export)
    assert media.metadata_carried(src_tags, media.media_tags(dst)) == []


def test_manifest_records_the_original_of_a_normalised_file(tmp_path):
    from videoredact.core.audit import build_manifest
    from videoredact.core.model import MediaInfo, Project
    m = MediaInfo(path=str(tmp_path / "phone_CFR.mp4"), fps=30, frame_count=120, has_video=True, vfr=True,
                  avg_fps=29.1, original_path=str(tmp_path / "phone.mp4"), original_sha256="ab" * 32, normalized=True)
    p = Project(media=m)
    man = build_manifest(p, {"output": "x"}, {"output": "x"})
    s = man["source"]
    assert s["variable_frame_rate_detected"] is True and s["normalized_from_vfr"] is True
    assert s["original_path"].endswith("phone.mp4") and s["original_sha256"] == "ab" * 32
    f = tmp_path / "p.vrproj"
    p.save(f)
    q = Project.load(f)
    assert q.media.original_sha256 == "ab" * 32 and q.media.normalized is True and q.media.vfr is True
    # projects from before these fields existed still load
    d = json.loads(f.read_text(encoding="utf-8"))
    for k in ("vfr", "avg_fps", "original_path", "original_sha256", "normalized"):
        del d["media"][k]
    f.write_text(json.dumps(d), encoding="utf-8")
    assert Project.load(f).media.normalized is False


def test_batch_normalises_a_vfr_file_before_detecting(tmp_path, vfr_clip, monkeypatch):
    from videoredact.core.batch import BatchOptions, run_batch
    import videoredact.core.batch as batch
    # no detector needed for this test: the auto-detect call is stubbed
    monkeypatch.setattr("videoredact.vision.auto_detect.run_auto_detect", lambda r, o, progress=None, cancel=None: [])
    out_dir = tmp_path / "out"
    res = run_batch(BatchOptions(files=[str(vfr_clip)], out_dir=str(out_dir), detect_labels=["face"], export=False))
    assert len(res) == 1 and res[0].ok, res[0].error
    cfr = out_dir / "phone_CFR.mp4"
    assert cfr.exists() and frame_timing(cfr)["vfr"] is False
    from videoredact.core.model import Project
    proj = Project.load(res[0].project)
    assert proj.media.path == str(cfr) and proj.media.original_path == str(vfr_clip) and proj.media.normalized
    assert any(e.action == "media_normalized" for e in proj.audit_log)
    assert len(proj.media.original_sha256) == 64 and proj.media.sha256 != proj.media.original_sha256
