"""Export writes atomically (no half-written output under the final name) and proves that
none of the source's metadata tags were carried into the output."""
from pathlib import Path

import pytest

from videoredact.core import media
from videoredact.core.export import ExportCancelled, ExportOptions, export_project, partial_path
from videoredact.core.media import media_tags, metadata_carried, probe
from videoredact.core.model import BBox, Project, Span, VideoTrack

SAMPLE = Path("tests/data/sample.mp4")

BANNER = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'clip.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 1
    compatible_brands: isommp42
    creation_time   : 2026-10-04T18:22:11.000000Z
    com.apple.quicktime.location.ISO6709: +33.4484-112.0740+331.000/
    comment         : Axon Body 3 X81234567
  Duration: 00:00:45.00, start: 0.000000, bitrate: 8000 kb/s
  Stream #0:0[0x1](eng): Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1280x720, 7800 kb/s, 30 fps, 30 tbr, 90k tbn (default)
      Metadata:
        creation_time   : 2026-10-04T18:22:11.000000Z
        handler_name    : VideoHandler
        vendor_id       : [0][0][0][0]
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 128 kb/s (default)
      Metadata:
        handler_name    : SoundHandler
At least one output file must be specified
"""


def test_media_tags_parses_format_and_stream_sections(monkeypatch):
    class R:
        stderr = BANNER
    monkeypatch.setattr(media, "run_ffmpeg", lambda *a, **k: R())
    tags = media_tags("clip.mp4")
    assert tags["format"]["comment"] == "Axon Body 3 X81234567"
    assert tags["format"]["com.apple.quicktime.location.ISO6709"] == "+33.4484-112.0740+331.000/"
    assert tags["stream0"] == {"creation_time": "2026-10-04T18:22:11.000000Z", "handler_name": "VideoHandler",
                               "vendor_id": "[0][0][0][0]"}
    assert tags["stream1"] == {"handler_name": "SoundHandler"}
    assert "Duration" not in tags["format"]          # the Duration line ends the metadata block


def test_metadata_carried_reports_only_identifying_values():
    src = {"format": {"comment": "Axon Body 3 X81234567", "encoder": "Lavf61", "major_brand": "mp42",
                      "creation_time": "2026-10-04T18:22:11Z"},
           "stream0": {"handler_name": "VideoHandler"}}
    clean = {"format": {"encoder": "Lavf61", "major_brand": "isom"}, "stream0": {"handler_name": "VideoHandler"}}
    assert metadata_carried(src, clean) == []
    leaky = {"format": {"comment": "Axon Body 3 X81234567", "creation_time": "2026-10-04T18:22:11Z"}}
    assert metadata_carried(src, leaky) == ["format.comment", "format.creation_time"]


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample clip missing (run scripts/make_sample.py)")
def test_export_is_atomic_and_carries_no_source_metadata(tmp_path):
    proj = Project(media=probe(str(SAMPLE), compute_hash=True), author="test")
    proj.add_video_track(VideoTrack(label="face", spans=[Span(0, 20, {0: BBox(0.2, 0.2, 0.2, 0.2)})]))
    out = tmp_path / "clip_redacted.mp4"
    part = partial_path(out)
    assert part.parent == out.parent and part.suffix == out.suffix and part != out

    # cancelled mid-encode: no output, no partial file left behind
    seen = []

    def cancel():
        return any(m.startswith("Encoding frame") for m in seen)

    def progress(p, m):
        seen.append(m)

    with pytest.raises(ExportCancelled):
        export_project(proj, ExportOptions(str(out), end_s=2.0, write_report=False, write_project=False,
                                           write_audit=False), progress=progress, cancel=cancel)
    assert not out.exists() and not part.exists()
    assert not list(tmp_path.glob("*.part*"))

    # a normal export: output present, partial gone, manifest proves no source tags were copied
    res = export_project(proj, ExportOptions(str(out), end_s=1.0, write_report=False, write_project=False,
                                             write_audit=True), progress=lambda p, m: None)
    assert out.exists() and not part.exists()
    assert res["source_metadata_carried"] == []
    assert isinstance(res["output_metadata"], dict)
    import json
    m = json.loads(Path(res["audit_json"]).read_text(encoding="utf-8"))
    assert m["output"]["source_metadata_carried"] == []
    assert m["output"]["metadata_tags"] == res["output_metadata"]
    assert all(x["matches_registry"] in (True, None) for x in m["models"])
