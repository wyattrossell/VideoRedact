# Legal, records and evidence-integrity drivers (US, Oct 2026)

Condensed research notes. Items marked *unverified* were not confirmed from a primary source.

## What gets redacted
- Faces/identity of uninvolved persons, minors, victims (DV/SA), witnesses, undercover officers, informants. Illinois requires removing identification of anyone not the officer, subject, or directly involved (50 ILCS 706/10-20(b)). Oregon requires all faces unidentifiable before disclosure (ORS 192.345(40)). NJ AG Directive 2022-1 s9.3 tags recordings showing victims, children, undercover/CIs, tactical info and **police computer screens with confidential data**.
- Locations: private residence interiors, medical/mental-health/social-service facilities (FL 119.071(2)(l) confidential; WA RCW 42.56.240(14) "highly offensive" presumption plus PHI, intimate images, minors, deceased, DV/SA victims). Texas bars release of any portion recorded in a "private space" without written authorization (Occ. Code 1701.661). Colorado requires nudity, gruesome injury and home/treatment interiors blurred (CRS 24-31-902).
- Spoken and on-screen PII: Spokane's guide lists spoken/visible DOB, phone, address, email, SSN, DL, card/account numbers, juvenile names, medical statements, each with an RCW cite. MDT screens are a standard full-frame-blackout case (Seattle).
- California Gov. Code 7923.625: critical-incident video within 45 days; redaction by blurring/distorting "shall not interfere with the viewer's ability to fully, completely, and accurately comprehend the events"; the recording "shall not otherwise be edited or altered". Penal 832.7(b)(6) limits what may be redacted.

## What records officers must document
- Federal FOIA 5 USC 552(b): amount deleted and the exemption indicated at the place of deletion. WA RCW 42.56.210(3): specific exemption + brief explanation of how it applies. CA 7923.625(b): written basis. Spokane's three-column table (material withheld / explanation / statutory basis) is the de facto log format; **no national report standard exists**.
- Industry norm: Axon exports an agency-configured exemption log; Veritone a zip with audit log. SF DPA rejected Premiere/Final Cut for lacking an audit log; its workflow: AI pre-mark -> staff review -> attorney approval -> redacted copy retained 3 years.
- Originals: redact editable copies, never the original; WA requires >= 60-day BWC retention; NJ s9.1 requires documenting every access/copy/dissemination.

## Evidence integrity
- SWGDE 18-V-001 v1.1 (2024): work on a copy; hash original and copy; contemporaneous notes of the order and settings of processes so another analyst can replicate. SHA-2 recommended. Hashes support authentication under FRE 901(b)(4) and 902(13)-(14); duplicates admissible under FRE 1003.
- CJIS Security Policy v6.0 (Dec 2024, NIST 800-53 aligned): audit events, unique user identification, MFA, FIPS crypto; P1 controls now, P2-P4 by Oct 2027. BWC video is not per se CJI, but footage showing NCIC/CHRI screens is; offline-only operation removes the cloud addendum burden.
- Open source: FBI certifies no products; no LE-specific OSS policy found (*unverified gap*). CISA's 2026 OSS guidance expects trustworthiness evaluation, inventory, SBOMs.

## Court / discovery
- Defense gets unredacted footage in discovery (Brady); redaction is for public copies. No reported decision rejecting blur vs box was found (*unverified*).
- **Reversibility is real**: Hill et al. 2016 recover text from mosaic/blur; McPherson et al. 2016 defeat pixelation/blur for faces with neural nets; Positive Security 2022 reverses *video* pixelation using camera motion across frames and recommends an opaque single-colour box. Gait/clothing/tattoos still re-identify blurred people.
- Tension: CA's comprehension language favours targeted boxes over whole-frame blackout and forbids trimming/re-timing.

## Workload
- Seattle 2024: 10 staff-minutes per video-minute per tracked object; audio-only / full-frame ~1:1.
- Backlogs: NYPD 133 business days average; Dallas ~4 months; Tucson 6-8 months; Oakland >= 45 days for < 30 min of video.
- Cost recovery is per staff-minute (Seattle $0.80/min; Texas $10/recording + $1/min, *unverified*): built-in time tracking matters.

## Prioritized requirements (effort S/M/L)
1. Per-redaction metadata: category, exemption citation (state pick-lists), free-text reason; export exemption log CSV + PDF (M)
2. SHA-256 of source at import and every export; re-verify source untouched at export; sidecar manifest with tool version and settings (S) -- hashes exist; manifest/settings todo
3. Non-destructive project separate from source; source read-only; outputs named as release copies (S)
4. Opaque box default; blur/pixelate behind a warning with enforced minimum block size; irreversible mode auto-selected for text (plates, screens, documents) (S)
5. Full-frame blackout + audio drop for a time range as a one-click primitive (S)
6. Audio mute/bleep with waveform view; STT to find names/DOB/SSN (exists) (M)
7. Local tamper-evident audit log (Windows user, timestamp, action, hash-chained) bundled with exports (M)
8. Review/approval state (draft -> reviewed -> approved, reviewer identity, lock after approval) (M)
9. Comprehension guardrail: no trimming/speed changes by default; if a segment is cut, insert a slate and log it (S)
10. Built-in time tracking per request (S)
11. Offline face/plate/screen detection with "redact everyone except selected" and human confirmation (L)
12. Trust posture: no network calls except the opt-in update check, SBOM, signed builds, one-page CJIS/records statement for agency IT (S)
