# Licences of everything VideoRedact ships, uses in development, or evaluated

Checked 2026-10-15 against the installed packages (`importlib.metadata`) and the upstream
repositories linked below. Project rule (NOTES.md "Decisions"): **permissive only** for anything
shipped: MIT, BSD, Apache-2.0, or LGPL used as an unmodified dynamically-linked library. Anything
else is listed here as **FLAGGED** with what it would take to use it.

Legend: ✅ permissive, shipped · 🟡 permissive with a condition we meet · 🔴 FLAGGED (not shipped,
or must be replaced before wide distribution) · ⚪ dev/evaluation only, not in the installer.

## 1. Models shipped in the installer

| Model | File | Use | Licence | Training data | Status |
|---|---|---|---|---|---|
| YuNet (OpenCV Zoo, 2023mar) | `face_detection_yunet_2023mar.onnx` | face detector (default) | MIT | WIDER FACE (academic dataset; model weights released under MIT by the authors) | ✅ |
| CenterFace | `centerface.onnx` | face detector (option) | MIT | WIDER FACE | ✅ |
| YOLOX-tiny / YOLOX-s (Megvii) | `yolox_tiny.onnx`, `yolox_s.onnx` | screens, phones, documents, people | Apache-2.0 | COCO (annotations CC BY 4.0; images Flickr terms). Standard practice treats the weights as Apache-2.0. | ✅ |
| YOLOX-tiny fine-tuned on our footage (optional) | `yolox_tiny_doc.onnx` | documents / notepads / paper, licence plates | Apache-2.0 (YOLOX code and COCO-pretrained starting weights); training data is our own footage, pseudo-labelled by OWLv2 (Apache-2.0) and reviewed locally | HPD body-cam frames (CJIS, never leave the machine); no public dataset | 🟡 only when present |
| ViTTrack (OpenCV Zoo) | `object_tracking_vittrack_2023sep.onnx` | single-object tracker | Apache-2.0 | GOT-10k / LaSOT / TrackingNet (academic) | ✅ |
| faster-whisper `small` (Systran CTranslate2 conversion of OpenAI Whisper) | `models/whisper/...` | speech-to-text | MIT (both OpenAI's weights and Systran's conversion) | OpenAI's undisclosed web corpus | ✅ |
| Silero VAD (inside faster-whisper, off by default) | bundled with faster-whisper | voice activity | MIT | – | ✅ |

Every model file's SHA-256 and licence is printed in each export's audit manifest and PDF
report (Phase 4), so the exact weights behind a redaction are always on record.

## 2. Libraries shipped in the installer

| Package | Version | Licence | Notes | Status |
|---|---|---|---|---|
| PySide6 / shiboken6 (Qt 6) | 6.11.2 | LGPL-3.0 (or GPL/commercial) | Used as unmodified DLLs in a onedir build; users can replace the Qt DLLs, which satisfies LGPL §4. We do not modify Qt. | 🟡 |
| numpy | 2.5.3 | BSD-3-Clause (+ bundled 0BSD/MIT/Zlib/CC0 parts) | | ✅ |
| opencv-contrib-python | 5.0.0.93 | Apache-2.0 | The wheel excludes the patented/"nonfree" modules; we use core, dnn, tracking (KCF/CSRT), imgproc. | ✅ |
| onnxruntime | 1.30.0 | MIT | Windows ETW telemetry disabled in code (Phase 4). | ✅ |
| onnx | 1.23.2 | Apache-2.0 | | ✅ |
| openvino | 2026.4.1 | Apache-2.0 | Intel iGPU/NPU tier. | ✅ |
| openvino-telemetry | 2025.2.0 | Apache-2.0 | **Excluded from the installer and blocked at import** (posts to Google Analytics by default; see docs/CJIS_COMPLIANCE.md). Licence fine, behaviour not. | 🔴 excluded |
| faster-whisper | 1.2.1 | MIT | | ✅ |
| ctranslate2 | 4.8.2 | MIT | | ✅ |
| huggingface-hub, tokenizers, hf-xet | 0.x / 0.23 / 1.6 | Apache-2.0 | Only used to locate the bundled Whisper model; forced offline. | ✅ |
| trackers (Roboflow) | 2.6.1 | Apache-2.0 | BoT-SORT association. | ✅ |
| supervision (Roboflow) | 0.30.9 | MIT | Dependency of trackers. | ✅ |
| scipy | 1.18.1 | BSD-3-Clause | | ✅ |
| soundfile | 0.14.0 | BSD-3-Clause | Bundles **libsndfile (LGPL-2.1)** as a DLL, unmodified. | 🟡 |
| cffi | 2.1.1 | MIT-0 | | ✅ |
| reportlab | 5.0.1 | BSD-3-Clause | PDF reports. | ✅ |
| pillow | 12.3.0 | MIT-CMU | | ✅ |
| requests, urllib3, certifi, idna, charset-normalizer | – | Apache-2.0 / MIT / MPL-2.0 / BSD / MIT | Transitive (huggingface-hub). MPL-2.0 (certifi) is file-level copyleft on certifi's own files only; unmodified. | ✅ |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | Transitive, unmodified. | ✅ |
| filelock, fsspec, packaging, PyYAML, typing-extensions, flatbuffers, protobuf, sympy, mpmath, coloredlogs, humanfriendly, pyreadline3, colorama, six, python-dateutil | – | MIT / BSD / Apache-2.0 | Transitive runtime deps of the above. | ✅ |
| **FFmpeg** (`bin/ffmpeg.exe`, gyan.dev "essentials" 7.1) | 7.1 | **GPL-3.0** build (includes libx264, GPL) | Invoked as a separate process with command-line arguments, which most readings consider "mere aggregation"; still, a GPL binary inside the installer is a distribution question the agency should not have to answer. **Swap for an LGPL build (e.g. a `--disable-gpl` BtbN/gyan "lgpl" build with libopenh264 or the Windows Media Foundation H.264 encoder) before wide distribution.** Tracked in ROADMAP. | 🔴 replace |
| imageio-ffmpeg | 0.6.0 | BSD-2-Clause | Dev-only fallback that also carries a GPL FFmpeg binary; **excluded from the installer** (`installer/videoredact.spec`). | ⚪ |
| PyInstaller | 6.22.3 | GPL-2.0-or-later **with the bootloader exception** | The exception explicitly allows bundling applications of any licence. Build tool only. | 🟡 |
| pyinstaller-hooks-contrib | 2026.8 | Apache-2.0 / GPL-2.0 (same exception) | Build tool only. | 🟡 |
| Inno Setup | 6 | Inno Setup licence (free for commercial and non-commercial use, redistribution of the installer allowed) | Build tool only. | ✅ |

## 3. Development / evaluation only (never in the installer)

| Component | Licence | Why it is listed | Status |
|---|---|---|---|
| SCRFD (InsightFace) code | MIT | Evaluated as a face detector in Phase 2 (`vision/detectors/scrfd.py`). | ⚪ code only |
| SCRFD / InsightFace **pretrained weights** (`det_10g.onnx` etc.) | **Non-commercial research only.** The InsightFace README (checked 2026-10-15): "The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only"; commercial use requires contacting InsightFace. | Scored below YuNet+multiscale on our frames anyway (69 % vs 96 %). **Not shipped, not used by the app.** The eval adapter stays for comparisons only. | 🔴 not shipped |
| Grounding DINO (open-vocabulary detector) | Apache-2.0 (code and weights; IDEA Research) | Evaluated for prompt-based detection; 9 s/frame on CPU. Parked. | ⚪ |
| PyTorch, transformers (eval venv) | BSD-3-Clause / Apache-2.0 | Only in `C:\dev\venv-eval` for evaluations and `train/`. | ⚪ |
| Label Studio | Apache-2.0 | Local labelling tool for the benchmark ground truth (`bench/label.ps1`). Runs on the labeller's machine; no footage leaves it. | ⚪ |
| CVAT (alternative labeller) | MIT | Not installed; listed in bench/README as an option. | ⚪ |

## 4. Candidates from the plan: verified terms (2026-10-15)

| Candidate | Licence as verified | Verdict for VideoRedact |
|---|---|---|
| **Ultralytics YOLO26 / YOLOE / YOLO11** code and weights | **AGPL-3.0** by default (ultralytics.com/license: "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default"). Distribution requires "publicly releasing the complete corresponding source code for the entire derivative work, including ... model weights", or an Ultralytics Enterprise licence. | 🔴 **Do not add.** VideoRedact is distributed as an installer to other agencies; AGPL would attach to the whole program. Our open-vocabulary need is covered by evaluation-only Grounding DINO today and RF-DETR / YOLOE-free alternatives tomorrow. `YOLO_OFFLINE=1` is set anyway as a guard (Phase 4). |
| **InsightFace / SCRFD weights** | Non-commercial research only (see section 3). | 🔴 Not shipped. |
| **RF-DETR** (Roboflow) | `rfdetr` package and the "Apache-designated" weights (Nano/Small/Medium/Base/Large): **Apache-2.0**. **RF-DETR-XL / 2XL and the `rfdetr_plus` extension: Roboflow "PML 1.0"**, a non-permissive platform licence. Pre-training data: DINOv2 backbone (Apache-2.0 weights) + Objects365/COCO (research datasets). | ✅ Apache sizes are acceptable as an ONNX export (needs the `onnx`/`onnxruntime` path only at runtime; torch stays in the eval venv). Avoid XL/2XL. Candidate for the open-vocabulary/"documents, screens" detector in a later phase, after a CPU speed check. |
| **Roboflow trackers / supervision** | Apache-2.0 / MIT | ✅ Already shipped (BoT-SORT). |
| **EgoBlur** (Meta: face + licence-plate detectors) | **Apache-2.0** for code and model weights ("The model is licensed under the Apache 2.0 license"). Models are TorchScript `.jit` (gen1/gen2). | ✅ Acceptable. Needs a one-time TorchScript → ONNX conversion in the eval venv (torch not shipped) and a CPU benchmark; the licence-plate detector fills a known gap (ROADMAP item 6). |
| **YuNet** (OpenCV Zoo) | MIT | ✅ Shipped, default. |
| **SAHI** (sliced inference) | MIT | ✅ Fine, but our `TiledDetector` already does the same; not needed. |
| **SAM 3** (Meta) | **"SAM License"** (repository LICENSE, checked 2026-10-15). Summary: non-exclusive, worldwide, royalty-free licence to use, modify and distribute the "SAM Materials" (code and weights), commercial use allowed; **restrictions**: no reverse engineering of the materials, no use subject to ITAR or trade-control-prohibited end uses "including those related to military or warfare purposes, nuclear industries or applications, espionage, or the development or use of guns or illegal weapons", must comply with privacy and data-protection law, must acknowledge SAM in publications, licence terminates on patent litigation against Meta. No OSI approval; not GPL-style copyleft either. | 🟡 **Usable for law-enforcement redaction** as far as the text goes (none of the prohibited end uses apply to evidence redaction; the privacy clause is one we already honour), but it is a bespoke licence that an agency's counsel would have to read. Keep it **optional and off by default** (as the Phase 2 plan said) and ship it only if it is a separately downloadable add-on with the licence text shown. Independently of licence, SAM 3 is far too slow on the 8-core/no-GPU target (ROADMAP: parked). |
| **deface** (ORB-HD) | MIT (code); uses CenterFace weights (MIT) | ✅ Reference implementation only; nothing imported. |
| **ONNX Runtime EPs, TensorRT, OpenVINO** | MIT / NVIDIA TensorRT licence (proprietary runtime, redistributable per NVIDIA's terms; CUDA EULA) / Apache-2.0 | OpenVINO shipped. TensorRT/CUDA tier parked: would require NVIDIA's redistribution terms and 1+ GB of DLLs; fleet has no NVIDIA GPUs. |
| **PyAV** | BSD-3-Clause (wrapper) over FFmpeg libs (LGPL/GPL per build) | ⚪ Not used; our decode path uses OpenCV + FFmpeg pipes. Same FFmpeg-build question as above if adopted. |

## 5. Action items

1. 🔴 **FFmpeg GPL build → LGPL build** before distributing beyond HPD (ROADMAP "Packaging").
   Until then the installer contains a GPL binary used as a separate process.
2. 🟡 Keep PySide6/Qt and libsndfile as unmodified DLLs (they are); mention LGPL re-linking in the
   installer's licence page (`installer/*.iss`, to do).
3. 🔴 Never add Ultralytics or InsightFace weights to `requirements.txt` or `models.py`;
   `tests/test_licenses.py` fails if either name appears there.
4. Candidates cleared for future phases: EgoBlur (Apache-2.0, plates), RF-DETR Apache sizes.
   SAM 3 only as an optional add-on with its licence text shown.
