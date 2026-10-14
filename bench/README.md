# Benchmark and labeling workflow

Everything here runs locally. Footage, extracted frames and labels never leave
the machine and are git-ignored (CJIS). Only `bench/results/*.json` and
`bench/labels/classes.txt` are committed.

## 1. Test clips
```powershell
C:\dev\venv-videoredact\Scripts\python bench\make_clips.py "samples\*.MP4" --every 300 --length 45 --max 8
```
Cuts 45-second clips every 5 minutes from each recording into `bench\clips\`.
Add any short clips of your own there too (mp4/mov, any resolution). Aim for
20-50 clips covering the situations that matter: traffic stops (plates, MDT
screens), interviews (faces, documents, ID cards), scenes with crowds, night
footage, phones held up to the camera.

## 2. Frames to label
```powershell
C:\dev\venv-videoredact\Scripts\python bench\extract_frames.py --every 30
```
Writes one JPEG per 30 frames (1 per second at 30 fps) to `bench\frames\<clip>\`.
The benchmark scores exactly these frames, so every extracted frame should be
reviewed, including ones with nothing sensitive (submit them empty).

## 3. Pre-annotate (optional but saves most of the work)
```powershell
C:\dev\venv-videoredact\Scripts\python bench\preannotate.py
```
Runs the current detectors over the frames and writes
`bench\labelstudio\tasks.json` with their boxes as *predictions*. You correct
them instead of drawing from scratch. Predictions are never used as ground
truth: only what you submit counts.

## 4. Label in Label Studio (local)
One-time install (separate venv, keeps the app venv clean):
```powershell
python -m venv C:\dev\venv-labelstudio
C:\dev\venv-labelstudio\Scripts\pip install label-studio
```
Start it:
```powershell
.\bench\label.ps1
```
Then in the browser (http://localhost:8080):
1. Create an account (stored locally in `bench\labelstudio\data`, nothing online).
2. Create Project -> *Labeling Setup* -> *Custom template* -> *Code* -> paste `bench\labeling_config.xml`.
3. *Import* -> upload `bench\labelstudio\tasks.json`. Images load from `bench\frames` through local-file serving.
   (If you skip pre-annotation: Settings -> Cloud Storage -> *Add Source Storage* -> Local files,
   path `bench\frames`, tick "Treat every bucket object as a source file", Sync.)
4. Label. Hotkeys 1-6 pick the class, drag to draw, Ctrl+Enter submits. Fix the predicted boxes,
   add missed ones, delete wrong ones. Classes:
   `face` (whole head when visible), `screen` (MDT/laptop/TV glass), `phone_screen`,
   `document` (papers, notepads, forms), `plate`, `id_card`.
5. Export -> JSON (not JSON-MIN) -> save as `bench\labelstudio\export.json`.

Convert to the benchmark's YOLO layout:
```powershell
C:\dev\venv-videoredact\Scripts\python bench\ls_to_yolo.py bench\labelstudio\export.json
```
CVAT is also fine if you prefer it (run it locally in Docker); export *YOLO 1.1*
or *COCO 1.0* and drop the files under `bench\labels\<clip_stem>\` with the
same `<clip_stem>_f000123` naming, or `bench\labels\<clip_stem>.json` for COCO.

## 5. Run the benchmark
```powershell
C:\dev\venv-videoredact\Scripts\python bench\run_bench.py --out bench\results\baseline.json
```
Per clip and aggregated it reports:
- **speed**: fps and seconds for decode, detect, track (carry), redact, encode, plus
  single-process and parallel auto-detect wall clock and export wall clock;
- **accuracy** on labelled frames, per class: `recall@90` (ground-truth boxes at least
  90 % covered by redaction), `precision` (predicted boxes that mostly lie on some
  ground-truth box; low = over-redaction), and **`leak_rate`**: the share of labelled
  frames in which any sensitive region is less than 90 % covered. Leak rate is the
  number to drive to zero.

Re-run after every pipeline change and commit the JSON next to `baseline.json`
(e.g. `bench\results\2026-10-20_openvino.json`) so gains are measured, not claimed.
