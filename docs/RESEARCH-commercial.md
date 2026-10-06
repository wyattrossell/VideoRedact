# Commercial redaction products: what they do that we don't (Oct 2026)

Condensed from a web research pass. Vendor claims are marked as such.

## Framing numbers
- Washington agencies' stopwatch studies: **10-11 staff-minutes per minute of raw footage per tracked person/object** (Seattle 2024, Spokane 2023, Thurston 2022); audio-only or full-frame blackout is ~1:1.
- Phoenix PD (2025): ~4 staff-hours per video hour; now releases a medium blur over the whole video first and charges extra for targeted redaction.
- Spokane PD workflow: Evidence.com -> download -> CaseGuard Studio -> frame-by-frame passes -> slow replay QA -> export -> deliver by link/thumb drive.

## Products
**Axon Redaction Studio / Redaction Assistant** (browser, inside Axon Evidence; de facto default for Axon agencies). Seven detectors (heads, plates, screens/MDTs, phones, notebooks, ID documents, audio PII via transcript); ~10 min for a 2-hour video (vendor); bulk up to 20 files. Reviewer walks a *Redacted Objects* panel (jump-to, exemption code, delete) and a colour-coded timeline (manual / tracker / audio / AI). Tools: rectangle/ellipse, Object Tracker (reposition -> re-tracks forward), *Spray Paint* (hold-and-drag along the object at half speed), inverted masks, per-segment blur level, audio mute/bleep from waveform. Export inherits case IDs; **exemption log** downloads alongside. Keyboard map: Space; 1/2/4 speeds; A/D frame step (hold = half speed); Q/E +/-2 s; W/S resize mask; arrows nudge; [ ] trim segment to playhead; F / Shift+F next/prev mask; hold M to lay audio mask; +/- zoom; Ctrl+Z. Unique-people thumbnail panel (2024). Weak: cloud/Chrome only, lock-in, decode failures on corrupt files.

**Veritone Redact** (AWS GovCloud SaaS, priced by processing hours, $2.4k/24 h and up). Detections listed chronologically with per-category show/hide; transcript synced for keyword search and click-to-redact; file status tags; comments; exports redacted file + audit log (Excel). SF DPA: detects shapes but does not track individuals; added attorney sign-off and 3-year retention of redacted copies.

**Motorola CommandCentral Evidence / WatchGuard**. AI catalogs people, faces, plates, vehicles, screens, documents; transcript filter by category (names, DOB, phone). Tools: Smart Tracking, manual, **full-frame redaction**, audio mute/beep on waveform. **Auto-pause when a track slips off the object**; named layers; multiple projects per source; duplicate project for another audience; export preview; trim; **redaction reasons per mask**; uneditable audit log; transcripts must be marked verified; bookmarks with comments. Their own line: "redaction is only as good as one missed frame".

**Genetec Clearance**. Auto face detection -> **thumbnail grid of each detected face ("Person #01…")** -> mask one or **Mask all**; black or blur (3 levels); hold-to-track at 0.1x-10x; start/end mask at current time; timeline thumbnails. Weak: audio is all-or-nothing.

**CaseGuard Studio** (on-prem Windows, offline; $99-$329/mo; closest analogue to us). 12 video PII categories, selective keep-visible, transcription in 100+ languages with 33 audio PII categories, voice anonymization, OCR/PDF, **bulk 1000+ files, overnight scheduling, watch folder**, TeamSpace approvals, **redaction reasons aligned to state law**, tamper-evident audit trail, metadata scrub. Complaints: **no re-identification when a person leaves and re-enters** (we already do this), needs a GPU, usage caps.

**Secure Redact (Pimloc)**. Credits per video-minute (~$6-8/min top-up: punishing for 40-min BWC files). Faces/heads, plates, screens; Review Mode with thumbnails and track editing; **audio NER (names, locations, dates) with filter-by-category -> redact all**; transcript export.

**Sighthound Redactor** (desktop/server, offline, CPU fallback; $2.5k/yr seat). Regex transcript search, speaker filter, **speaker diarization: mute one voice only**; mute/beep/scramble; **built-in work-time logging per session**; audit archive with host/user manifest and **SHA-256 of the audit log**.

**Brighter AI** (synthetic face replacement, GDPR market), **Facit Identity Cloak** (on-prem UK; layered static background blur, in-app trim), **Suspect Technologies** (Azure Gov / on-prem; batch; beep/silence/custom tone), **YouTube Studio blur** (face grid -> click to blur for whole video).

## Agency guidance takeaways
- WA RCW 42.56.240(14): agencies may charge redaction *time* (not technology) -> a tool that logs reviewer time per request is directly useful for cost recovery.
- TX Occ. Code 1701.661; CA AB 748/SB 1421: critical-incident video within 45 days; redaction must not interfere with comprehension.
- DOJ OIP: document exemptions per redaction.

## 15 highest-value gaps for VideoRedact (effort S/M/L)
1. Exemption code per mask/audio segment + exportable exemption log (S)
2. Redacted-objects review queue with jump-to, category toggles, reviewed flags, "mark all verified" (M)
3. Unique-person thumbnail gallery; click a face -> redact all appearances; Mask all (L)
4. Track-loss auto-pause / missed-frame alerts (M)  -- partly covered by the post-export verification pass (v0.1.5)
5. Full-frame blur/blackout segments + inverted mask (S)
6. NER audio PII with filter-by-category -> redact all (M)
7. Speaker-diarized audio redaction (M)
8. Axon-style shortcut set (S)
9. Spray-paint / live-track manual mode at 0.1-10x (M)
10. In/out trim before detection + clip-only export (S)
11. Multi-file batch queue with presets, overnight schedule, watch folder (M)
12. Per-request work-time log (S)
13. Bookmarks/comments and per-mask notes; duplicate project for another audience (S)
14. Redaction templates/presets by request type and statute (S)
15. Robust decode of corrupt files, output metadata scrub, audit-log self-hash + host/user manifest, verify-before-release sign-off (M)

Sources: Axon help/blog pages, Veritone AWS listing and SF DPA deck, Motorola factsheet, Genetec Clearance help, CaseGuard site + Capterra, Secure Redact site, Sighthound 7.3 notes, Seattle/Spokane/Thurston cost studies, Phoenix PD newsroom, DOJ OIP best practices, Focal Forensics comparison.
