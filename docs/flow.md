# Execution Flow

Documents how execution actually moves between modules — what calls what, in what order — for this system's non-obvious runtime paths (Streamlit's rerun model, subprocess management, session-state persistence). Not a full API reference; a map of the paths that were non-trivial to get right.

---

## Flow 1: Live facial prediction cycle (`emotion_analyzer/webapp/app.js` → `api/main.py`)

Triggered every 1000ms by `setInterval(sendFrame, 1000)`, independent of any server-side loop.

1. `captureFrame()` — draws the current `<video>` frame to a `<canvas>`, returns `canvas.toDataURL('image/jpeg', 0.8)` as a base64 data URL.
2. `fetch(API_URL, {...})` — POSTs `{ image: dataUrl }` as JSON to `/predict` (this exact request/response shape was the original prototype's bug — it POSTed JSON against an endpoint expecting multipart form data; corrected here from the start).
3. `_decode_base64_image()` (`api/main.py`) — strips the `data:image/jpeg;base64,` prefix, decodes to a BGR `numpy` array via `cv2.imdecode`.
4. `predictor.predict_image(bgr)` (`inference/predictor.py`):
   a. `detect_faces()` — `cv2.FaceDetectorYN.detect()` (decisions.md D-005), wrapped in `try/except cv2.error` to skip a single bad frame rather than 500 the request on YuNet's occasional shape-mismatch assertion on a frame-size change.
   b. Each detected box padded ~15% per side before cropping, closing the gap between YuNet's tight boxes and the looser crop margins in the training data.
   c. `_preprocess_face()` — BGR→RGB, resize to `IMG_SIZE`, `mobilenet_v2.preprocess_input`.
   d. `model.predict()` — returns per-class probabilities; top class + confidence + full distribution returned per face.
5. Response JSON returned; `renderPredictions()` (`app.js`) draws a bounding box + label directly onto the canvas over the last captured frame.

---

## Flow 2: Live calibration pairing (`bci_suite/pages/1_Calibration_Ground_Truth.py`)

The non-obvious part of this flow is that **Streamlit reruns the entire script top-to-bottom on every widget interaction** — there is no persistent loop. Everything below that needs to survive across reruns is deliberately kept in `st.session_state`, not a local variable.

1. **Start stream:** clicking "▶ Start replay stream" calls `stream_launcher.start_replay_stream()`, which launches `replay.py` via `subprocess.Popen` and returns immediately — the handle is stored in `st.session_state.stream_proc` specifically because a local variable would be discarded on the very next rerun.
2. **Connect:** clicking "🔌 Connect to EEG stream" constructs one `EEGStreamReader`, calls `.connect()` **once**, and stores the reader itself in `st.session_state.eeg_reader`. `.connect()` resolves the LSL stream, opens an inlet, and — critically — calls `inlet.info()` to fetch the *full* stream descriptor (not the lightweight one `resolve_byprop()` returns), which is what makes channel-label reading possible at all (decisions.md D-012).
3. **Keeping the buffer "live" without a background loop:** on *every* rerun thereafter — regardless of what triggered it — `reader.pull_samples(timeout=0.1)` runs near the top of the live-mode block. This is what makes the stream feel continuous: each rerun opportunistically drains whatever new LSL samples have arrived since the last one, appending them to the reader's internal `deque`.
4. **New-photo detection:** `st.camera_input()`'s returned value persists across reruns caused by *other* widgets (e.g., clicking "Connect" again does not clear an already-taken photo). Without an explicit check, the same photo would be re-processed and re-logged on every subsequent rerun. `hashlib.md5(img_file.getvalue())` is compared against `st.session_state.last_photo_hash` to detect a genuinely new snapshot before doing anything with it.
5. **Auto-pairing (the actual point of this page, decisions.md D-013):** on a new photo — `predictor.predict_image()` → facial label → `facial_label_to_va()` + `va_to_quadrant()` → quadrant. `reader.get_latest_window()` slices the most recent `WINDOW_SAMPLES` samples from the buffer. If both are available, `{window, quadrant, facial_label}` is appended to `st.session_state.calibration_pairs` and `last_photo_hash` is updated — no separate "log" button.
6. **Training:** once `len(calibration_pairs) >= k_target`, clicking "🚀 Train personalized model" calls `personalize.finetune_from_calibration()` synchronously (inside `st.spinner`, blocking the script during the fine-tune) — loads the generic checkpoint, fine-tunes for `FINETUNE_EPOCHS`, exports to `models/eeg/personalized/<name>.onnx` + a metadata sidecar.
7. **Stop:** `stream_launcher.stop_stream()` terminates the subprocess; `stream_proc` and `eeg_reader` are reset to `None` in session state, followed by an explicit `st.rerun()` so the UI reflects the cleared state immediately rather than waiting for the next natural interaction.

---

## Flow 3: Fusion prediction (`bci_suite/pages/2_Multimodal_Fusion_Demo.py`)

1. `get_model_options()` — lists `"Generic (...)"` plus every entry from `personalize.list_personalized_models()` (each a saved `*_metadata.json`).
2. `load_eeg_classifier(model_choice)` — branches to `EEGEmotionClassifier()` (generic, default paths) or one constructed with `PERSONALIZED_DIR / f"{model_choice}.onnx"` and its matching labels file.
3. EEG source branches identically to Flow 2 steps 1–3 if "Live stream" is selected (same reader-in-session-state, same per-rerun `pull_samples()` pattern), or to a `st.selectbox()` over `get_sample_eeg_windows()` — a cached, seeded sample of DREAMER windows — if "Sample from DREAMER" is selected. In the latter case, **the true label is deliberately not shown in the dropdown text**, only revealed afterward via an `st.expander()`, for the same reason as Flow 2 (decisions.md D-013): showing it first would make prediction meaningless.
4. Facial channel: identical `camera_input` → `predict_image()` → VA → quadrant path as elsewhere.
5. `facial_quadrant == eeg_quadrant` comparison drives the explicit agree/disagree message — not just decoration, this is the stated reason the fusion page exists (facial expression can be masked, EEG is harder to; agreement is corroborating evidence, disagreement is itself informative).
6. `fused_va` = simple midpoint average of the two channels' VA points; plotted via `plot_circumplex()` alongside the individual channel points.

---

## Flow 4: Offline facial training (`emotion_analyzer/emotion_analyzer/models/train.py`)

1. `build_manifest("train")` / `build_manifest("test")` — merges FER2013 + RAF-DB (both split-aware) with AffectNet (train-only, confidence-filtered via `relFCs`, see decisions.md D-004), then drops the `contempt` class (only AffectNet contributes it, and it has no clean held-out split here).
2. `train_test_split(..., stratify=label_idx)` — class-stratified train/val split (no participant/group concept applies to facial images the way it does to EEG trials).
3. `compute_class_weight(class_weight="balanced", ...)` — compensates for FER2013's `disgust` (~436 examples) and RAF-DB's `fear` (~281) being dwarfed by `happy` (~7000+).
4. **Phase 1:** `build_mobilenet_emotion_model(fine_tune_at=None)` — backbone frozen, head trained via Keras `.fit()` with `ModelCheckpoint`/`EarlyStopping`/`ReduceLROnPlateau`.
5. **Phase 2:** model rebuilt with `fine_tune_at=60`, `load_weights()` from phase 1's `best.keras`, fine-tuned at a much lower learning rate (1e-5).
6. `model.evaluate(test_ds)` — the only point the held-out test set is touched.
7. **Promotion (decisions.md D-007):** the run saves to its own `RUN_DIR = models/facial_emotion/run_<timestamp>/` unconditionally. `test_acc` is compared against `best_test_accuracy.txt`; only a new best is copied over the "production" `emotion_model.keras` / `labels.json` paths that `inference/predictor.py` actually loads. A regressed run is never promoted, regardless of how the rest of training looked.

---

## Flow 5: Offline EEG training (`bci_suite/bci_suite/eeg/train.py`)

1. `load_dreamer_windows()` — returns windows, quadrant labels, and a `groups` array (which participant each window came from).
2. `GroupShuffleSplit` (not a plain stratified split) — first splits off a test set by participant, then splits the remainder into train/val, also by participant. No participant's data crosses between splits (decisions.md D-009 — this replaced an earlier window-level split that leaked).
3. `compute_class_weight` — DREAMER's class balance is uneven (`LVLA` is roughly 6x rarer than `HVHA`).
4. Single-phase manual PyTorch training loop (not Keras-callback-based, since EEGNet here is plain `torch.nn.Module`): each epoch's `state_dict` is cloned and kept only if it beats the best validation accuracy seen so far — the manual equivalent of `ModelCheckpoint(save_best_only=True)`.
5. Final evaluation against the held-out **test** loader (participants never touched during training or validation).
6. `torch.onnx.export()`, wrapped in `try/except ModuleNotFoundError` to fall back from the newer dynamo-based exporter (needs the optional `onnxscript` package) to the legacy exporter if that dependency isn't present (decisions.md D-010).
7. **No promotion safeguard here** — unlike Flow 4, this script always overwrites `models/eeg/eegnet_dreamer.onnx` directly. The facial model's promotion-on-improvement pattern (decisions.md D-007) was motivated by an incident that happened specifically to the facial pipeline and has not been retrofitted here; a regressed EEG retrain would currently overwrite a better one silently (see README Roadmap).
