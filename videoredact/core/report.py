"""Redaction report (CSV + PDF) for discovery / chain of custody.

The report lists every redaction with time ranges, who created it and when,
the style applied, SHA-256 of the source and output files, and the audit
log. It deliberately contains no thumbnails of redacted content.
"""
from __future__ import annotations

import csv
from pathlib import Path

from videoredact import __version__
from .model import Project, AudioStyle, VideoStyle, Shape, as_enum


def fmt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def write_csv(project: Project, path: str | Path, result: dict | None = None) -> None:
    fps = project.media.fps or 0.0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "id", "label", "start", "end", "duration_s", "start_frame", "end_frame",
                    "style", "shape", "text", "reason", "source", "enabled", "created_by", "created_at"])
        for r in project.audio_redactions:
            w.writerow(["audio", r.id, "speech", fmt_time(r.start), fmt_time(r.end), f"{r.duration:.3f}",
                        "", "", as_enum(AudioStyle, r.style or project.default_audio_style).value, "", r.text, r.reason,
                        r.source, r.enabled, r.created_by, r.created_at])
        for t in project.video_tracks:
            for s in t.spans:
                st = s.start_frame / fps if fps else 0
                en = (s.end_frame + 1) / fps if fps else 0
                w.writerow(["video", t.id, t.label, fmt_time(st), fmt_time(en), f"{en - st:.3f}",
                            s.start_frame, s.end_frame, as_enum(VideoStyle, t.style or project.default_video_style).value,
                            as_enum(Shape, t.shape).value, "", t.note, t.source, t.enabled, t.created_by, t.created_at])
        w.writerow([])
        w.writerow(["source_file", project.media.path])
        w.writerow(["source_sha256", project.media.sha256])
        if result:
            w.writerow(["output_file", result.get("output", "")])
            w.writerow(["output_sha256", result.get("output_sha256", "")])
        w.writerow(["software", f"VideoRedact {__version__}"])


def write_pdf(project: Project, path: str | Path, result: dict | None = None) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak

    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, leading=10)
    mono = ParagraphStyle("mono", parent=styles["Normal"], fontName="Courier", fontSize=7.5, leading=9)
    doc = SimpleDocTemplate(str(path), pagesize=landscape(letter), leftMargin=0.5 * inch,
                            rightMargin=0.5 * inch, topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title="Redaction Report", author=project.author or "VideoRedact")
    story = []
    story.append(Paragraph("Media Redaction Report", styles["Title"]))
    meta = [
        ["Case number", project.case_number or "-"],
        ["Prepared by", project.author or "-"],
        ["Report generated", project.modified_at],
        ["Software", f"VideoRedact {__version__} (open source, MIT)"],
        ["Source file", Paragraph(project.media.path, mono)],
        ["Source SHA-256", Paragraph(project.media.sha256 or "-", mono)],
        ["Duration", fmt_time(project.media.duration)],
        ["Video", f"{project.media.width}x{project.media.height} @ {project.media.fps:.3f} fps, {project.media.frame_count} frames" if project.media.has_video else "none"],
        ["Audio", f"{project.media.sample_rate} Hz, {project.media.channels} ch ({project.media.audio_codec})" if project.media.has_audio else "none"],
        ["Transcription model", project.transcript_model or "-"],
    ]
    if result:
        meta += [["Output file", Paragraph(result.get("output", ""), mono)],
                 ["Output SHA-256", Paragraph(result.get("output_sha256", ""), mono)]]
        if result.get("clip"):
            a, b = result["clip"]
            meta.append(["Exported range", f"{fmt_time(a)} - {fmt_time(b) if b is not None else 'end of recording'} of the source"])
        vr = result.get("verify")
        if vr is not None:
            meta.append(["Output verification", f"{vr.frames_checked} frames re-scanned for {', '.join(vr.labels)}; "
                                                f"{len(vr.uncovered)} detection(s) not covered by a redaction"])
    if project.notes:
        meta.append(["Notes", Paragraph(project.notes, small)])
    t = Table(meta, colWidths=[1.6 * inch, 7.8 * inch])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
                           ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 8.5),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
    story += [t, Spacer(1, 12)]

    # ---- audio -----------------------------------------------------------
    story.append(Paragraph(f"Audio redactions ({len(project.audio_redactions)})", styles["Heading2"]))
    rows = [["#", "Start", "End", "Dur (s)", "Style", "Redacted words", "Reason", "Source", "By", "Created"]]
    for i, r in enumerate(project.audio_redactions, 1):
        rows.append([str(i), fmt_time(r.start), fmt_time(r.end), f"{r.duration:.2f}",
                     as_enum(AudioStyle, r.style or project.default_audio_style).value,
                     Paragraph(r.text or "(range)", small), Paragraph(r.reason or "", small), r.source,
                     Paragraph(r.created_by, small), Paragraph(r.created_at[:19].replace("T", " "), small)])
    if len(rows) == 1:
        rows.append(["-", "", "", "", "", "none", "", "", "", ""])
    at = Table(rows, repeatRows=1, colWidths=[0.3 * inch, 0.9 * inch, 0.9 * inch, 0.5 * inch, 0.6 * inch,
                                                 2.6 * inch, 1.3 * inch, 0.7 * inch, 0.8 * inch, 1.2 * inch])
    at.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.5),
                            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde3ea")),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7f9")]),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
    story += [at, Spacer(1, 12)]

    # ---- video -----------------------------------------------------------
    fps = project.media.fps or 0.0
    story.append(Paragraph(f"Video redactions ({len(project.video_tracks)} tracked regions)", styles["Heading2"]))
    rows = [["#", "Label", "Style", "Shape", "Visible ranges (start - end)", "Frames", "Source", "Note", "By", "Created"]]
    for i, tr in enumerate(project.video_tracks, 1):
        ranges = "; ".join(f"{fmt_time(s.start_frame / fps) if fps else s.start_frame} - "
                           f"{fmt_time((s.end_frame + 1) / fps) if fps else s.end_frame}" for s in tr.spans)
        rows.append([str(i), tr.label, as_enum(VideoStyle, tr.style or project.default_video_style).value,
                     as_enum(Shape, tr.shape).value,
                     Paragraph(ranges, small), str(tr.total_frames()), tr.source,
                     Paragraph(tr.note, small), Paragraph(tr.created_by, small),
                     Paragraph(tr.created_at[:19].replace("T", " "), small)])
    if len(rows) == 1:
        rows.append(["-", "none", "", "", "", "", "", "", "", ""])
    vt = Table(rows, repeatRows=1, colWidths=[0.3 * inch, 0.9 * inch, 0.6 * inch, 0.5 * inch, 3.0 * inch,
                                                 0.6 * inch, 0.6 * inch, 1.1 * inch, 0.8 * inch, 1.2 * inch])
    vt.setStyle(at.getStyle() if hasattr(at, "getStyle") else TableStyle([]))
    vt.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.5),
                            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde3ea")),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7f9")]),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
    story += [vt, Spacer(1, 12)]

    # ---- audit log -------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph(f"Audit log ({len(project.audit_log)} entries)", styles["Heading2"]))
    rows = [["Timestamp", "User", "Action", "Details"]]
    for e in project.audit_log:
        rows.append([e.ts[:19].replace("T", " "), Paragraph(e.user, small), e.action, Paragraph(e.details, small)])
    lt = Table(rows, repeatRows=1, colWidths=[1.4 * inch, 1.2 * inch, 1.8 * inch, 5.3 * inch])
    lt.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.5),
                            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde3ea")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey)]))
    story.append(lt)
    doc.build(story)
