# Engineering Decision Log

Each entry records a meaningful technical decision made during this project, why it was made, what alternatives existed, and what its known consequences are. This document answers *why*, not *what changed* — see `docs/PROJECT_NARRATIVE.md` for the latter. Entries are numbered chronologically by when the decision was made, not by importance.

---

## D-001: FER2013 + FER+ as the training set for the public model, restricted datasets for validation only

**Status: superseded by D-017 — see below. Left in place rather than deleted, per this log's own policy of recording supersession rather than rewriting history.**

**Phase:** 1 (Dataset Acquisition)
**Decision:** Train the public-facing facial model on FER2013 and FER+ only. Use AffectNet and RAF-DB (both research-only-licensed, request-gated datasets) for cross-dataset validation, never for training the shipped model.
**Reasoning:** AffectNet, RAF-DB, and DREAMER's terms restrict redistribution and, for some, commercial use of the data itself. FER2013 originates from the ICML 2013 challenge with no such restriction. Keeping the shipped model's training data unambiguous avoids a licensing question hanging over the one artifact meant to be public-facing.
**Consequences:** A smaller, noisier primary training set than combining everything would give, in exchange for a training provenance that's simple to state and defend.

## D-002: MobileNetV2 transfer learning over a CNN trained from scratch

**Phase:** 2 (Facial Model Architecture)
**Decision:** Fine-tune an ImageNet-pretrained MobileNetV2 rather than train a small custom CNN from scratch (as the original, pre-rewrite prototype had done).
**Reasoning:** The original prototype's from-scratch CNN never got past `model.summary()` — no `.fit()` call existed in the inherited notebook. Given a small-to-moderate dataset budget, transfer learning from ImageNet features reliably outperforms training a comparable-capacity network from scratch.
**Consequences:** Required a two-phase training procedure (frozen head, then fine-tune) rather than single-phase training, and a fixed 224×224 input size dictated by MobileNetV2's pretrained weights.

## D-003: RAF-DB mislabeling discovered via class-size fingerprinting, not documentation

**Phase:** 1 (Dataset Acquisition)
**Decision:** A dataset downloaded and labeled "ferplus" on Kaggle was identified as an unlabeled RAF-DB mirror, not Microsoft's actual FER+ (vote-distribution relabeling of FER2013), by comparing its numeric folders' per-class sample counts (12,271 total, matching RAF-DB's published train-set size exactly) against known distributions, rather than trusting the filename.
**Reasoning:** The folder was named `ferplus` and had a superficially plausible structure (numbered folders, image/label CSVs). Nothing about it self-identified as RAF-DB. The mismatch was only found by treating the filename as a claim to verify, not a fact — cross-referencing the exact per-class counts against RAF-DB's documented distribution (Happiness 4772, Fear 281, etc.) confirmed the match beyond coincidence.
**Consequences:** Avoided training against an incorrectly-assumed label mapping (Microsoft's FER+ format is vote-distribution across named emotions; RAF-DB's is a fixed 1–7 numeric code) that would have silently produced a broken model while appearing to train normally.

## D-004: AffectNet's CSV labels trusted over folder placement

**Phase:** 1 (Dataset Acquisition)
**Decision:** Load AffectNet labels from its `labels.csv` (which includes a `relFCs` confidence score), not from which folder each image happened to sit in.
**Reasoning:** Spot-checking the CSV showed folder placement and the CSV's `label` column disagreeing for some images (e.g., an image filed under `anger/` labeled `surprise` in the CSV) — evidence the folder reflects an earlier/different labeling pass than the CSV's, likely auto-relabeled. Trusting folder position would have silently mislabeled an unknown fraction of training data.
**Consequences:** Added a confidence-threshold filter (default 0.7) on `relFCs`, discarding low-confidence auto-labeled rows rather than using them uncritically.

## D-005: YuNet over Haar cascades or MediaPipe for face detection

**Phase:** 3 (Inference Pipeline)
**Decision:** Use OpenCV's `cv2.FaceDetectorYN` (YuNet) for face detection in the live inference pipeline, after both Haar cascades and MediaPipe failed for reasons unrelated to this project's code.
**Reasoning:** The installed `opencv-python` build's `cv2.CascadeClassifier` API was unavailable (a legacy-API removal in a newer OpenCV release). MediaPipe's Tasks API face detector crashed on macOS with a Metal/GPU service initialization error (`Check failed: service_ Service is unavailable`), confirmed to persist even with the CPU delegate explicitly forced — a known upstream platform issue, not something fixable from calling code. YuNet, OpenCV's own currently-maintained detector, required only downloading its ONNX model file and worked immediately.
**Consequences:** Three separate face-detection libraries were tried before finding one that worked on this environment — a genuine cost in build time, and a caution against assuming any single "standard" computer-vision library will behave identically across OS/version combinations.

## D-006: Augmentation-order bug — found, fixed, not treated as a tradeoff

**Phase:** 2 (Facial Model Training)
**Decision:** Moved data augmentation (`RandomBrightness`, `RandomContrast`, flip, rotation, zoom) to run on raw 0–255-scale images, *before* `mobilenet_v2.preprocess_input`'s normalization to `[-1, 1]`, rather than after it.
**Reasoning:** A retrain intended to improve accuracy via augmentation instead collapsed it from 61.2% to 29.7%, with phase-1 training never exceeding random chance. Root cause: `RandomBrightness`'s default `value_range=(0, 255)` was being applied to already-normalized `[-1, 1]` data, clipping nearly every pixel to an extreme and destroying the images the network trained on for the entire run. This was a genuine bug, not an accepted tradeoff — augmentation order is unambiguously wrong when applied after normalization for value-range-sensitive layers, and needed correcting, not documenting around.
**Consequences:** Corrected order retrained to 67.4%, beating the pre-augmentation baseline. This run also directly motivated D-007 (below) — the run that hit this bug had already overwritten the previous working model on disk.

## D-007: Model promotion is conditional, not automatic — every training run

**Phase:** 2 (Facial Model Training)
**Decision:** After the augmentation-order bug (D-006) silently overwrote a working 61.2%-accuracy model with a broken 29.7%-accuracy one, every subsequent training run saves to its own timestamped folder and only overwrites the "production" model path if its test accuracy beats the previously recorded best (tracked in a `best_test_accuracy.txt` marker file).
**Reasoning:** The original training script always saved to the same fixed path, meaning a regression — from a bug, from bad luck, from an ill-advised hyperparameter change — could silently replace a better model with a worse one, discoverable only after the fact. This is a correctness issue independent of how good any individual training run is.
**Consequences:** A worse run now sits harmlessly in its own folder instead of destroying the best-known model. Applied to the facial model's `train.py` after the incident; the same pattern was not (yet) retrofitted into the EEG training script, since no equivalent incident occurred there — see Roadmap.

## D-008: DREAMER over DEAP, after DEAP's official host was found discontinued

**Phase:** 4 (EEG Dataset Acquisition)
**Decision:** Use DREAMER (Zenodo-hosted, request-gated but stable) as the EEG-emotion dataset, after DEAP — the originally planned choice, notable for including synchronized facial video — was found to have a discontinued official host (a `scicrunch.org` resource listing marked it "no longer in service," documented mid-December 2025).
**Reasoning:** DEAP's synchronized facial video would have been a better fit for the calibration-page concept, but a dead host is not a workaround-able problem. DREAMER, while lacking synchronized video, offered a still-active access path and the same core valence/arousal/dominance labeling needed for the quadrant-classification task.
**Consequences:** The calibration page's "ground truth" concept had to be built around a live user's own facial expression paired against a *replayed* DREAMER window, rather than genuinely synchronized recorded facial+EEG data from the same original session — an accepted, documented limitation (see README).

## D-009: Participant-level split (GroupShuffleSplit) over window-level split for the EEG model

**Phase:** 4 (EEG Model Training)
**Decision:** Split DREAMER's windows into train/val/test by *participant*, using `sklearn.model_selection.GroupShuffleSplit`, rather than a plain random split across all windows.
**Reasoning:** Each ~60-second trial is chunked into many 4-second windows; a plain random split let windows from the same trial land in both train and test, allowing the model to partly "recognize" a trial it had already seen slices of rather than genuinely generalizing to a new person. This was directly visible as erratic validation accuracy (bouncing between 0.05 and 0.44 epoch to epoch) under the leaked split.
**Consequences:** Test accuracy actually *increased* slightly under the corrected, harder, leakage-free split (44.4% → 48.6%) — a useful reminder that a corrected methodology does not necessarily produce a lower number, only a trustworthy one.

## D-010: ONNX export fallback for a missing `onnxscript` dependency

**Phase:** 4 (EEG Model Export)
**Decision:** Wrapped the EEG model's ONNX export in a `try/except ModuleNotFoundError`, falling back from PyTorch's newer dynamo-based exporter (opset 18, requires the optional `onnxscript` package) to the legacy exporter (`dynamo=False`, opset 17) if the dependency is missing.
**Reasoning:** The first training run completed successfully but crashed during export specifically because `onnxscript` wasn't installed — a genuinely separate concern from model training, not worth letting invalidate a completed training run.
**Consequences:** Export succeeds either way; `onnxscript` was subsequently installed, so the fallback path exists as a safety net rather than the primary path in current runs.

## D-011: LSL as the streaming protocol, so real hardware and simulation share one code path

**Phase:** 5 (Live Streaming Architecture)
**Decision:** Built `replay.py` (simulates a headset by replaying DREAMER over LSL) and `stream.py` (a generic LSL reader) as two independent pieces, deliberately designed so neither knows or cares whether the LSL stream it's talking to originates from real hardware or a replayed recording.
**Reasoning:** The alternative — a custom in-process API for "fake" streaming during development, replaced later by a different real-hardware API — would mean the entire live pipeline (buffering, windowing, calibration, personalization) gets built and tested against one interface, then needs re-validating against a second, different one when real hardware arrives. Building on LSL from the start means that revalidation is unnecessary by construction.
**Consequences:** `brainflow_bridge.py` (real-headset support) was written and integrated without touching `stream.py`, `stream_launcher.py`, or any page logic at all — direct evidence the architectural bet paid off, though it remains unverified against actual physical hardware (see Limitations).

## D-012: Name-based channel matching over a raw channel-count check

**Phase:** 5 (Live Streaming Architecture)
**Decision:** `stream.py` matches incoming LSL channels to the model's required 14 electrode positions *by label* (when the stream publishes channel names), selecting and reordering just those 14 regardless of how many total channels the source has — rather than only checking that the channel count equals 14.
**Reasoning:** A raw count check would reject any headset with more than 14 channels even if all 14 needed positions are present among them (e.g., a 32-channel research cap). Name-based matching is real, not cosmetic, provided the stream publishes labels — both `replay.py` and `brainflow_bridge.py` were written to publish them for exactly this reason.
**Consequences:** Surfaced a genuine LSL API subtlety during testing: `pylsl.resolve_byprop()` returns a lightweight `StreamInfo` without custom metadata (like channel labels); the full metadata is only available after opening an inlet and calling `inlet.info()`. The first implementation read labels off the lightweight object and always found none, silently falling back to the count-only path — fixed once diagnosed. A headset genuinely missing a required electrode position is still, correctly, refused — this logic cannot invent missing physical channels.

## D-013: Calibration and fusion pages redesigned after a circular-logic flaw, found by user critique

**Phase:** 6 (Calibration/Fusion UX)
**Decision:** The original calibration and fusion pages displayed each EEG sample's true label directly in its selection dropdown (e.g., "participant 10, true label: Excited/Happy") before running any prediction. Redesigned so the true label is hidden until after prediction runs, revealed only via an explicit "reveal" expander.
**Reasoning:** Showing the answer before asking the question makes "prediction" meaningless — a "calibration" page that displays ground truth ahead of time isn't calibrating anything, it's a lookup table. This flaw was identified by the project owner reviewing the running app, not caught during development — an internal-review gap worth recording rather than omitting.
**Consequences:** Also motivated adding actual function to "calibration," which previously was a passive side-by-side display with no consequence: a session log of auto-paired (facial label, EEG prediction) records, explicitly framed as the kind of data a real system would accumulate to personalize the EEG model — the basis for the personalized fine-tuning feature (`personalize.py`) built immediately after.

## D-014: BrainFlow over per-vendor SDKs for real-headset support

**Phase:** 7 (Real Hardware Support)
**Decision:** Real EEG hardware support is built via BrainFlow (one library, one API, supporting OpenBCI Cyton/Ganglion, Muse 2/S, and other boards) rather than integrating separate vendor SDKs per headset brand.
**Reasoning:** A per-brand integration strategy would mean "connect your headset" only works for whichever single brand was implemented first, with each additional brand requiring its own bespoke code. BrainFlow's unified `BoardShim` API covers a meaningfully wide range of consumer/research boards through one bridge script.
**Consequences:** Not universal — ecosystems with proprietary, non-BrainFlow-supported APIs (notably Emotiv's official Cortex API) still require their own vendor software regardless. This is stated plainly rather than implied to be solved.

## D-015: Snapshot-based facial capture accepted as a platform limitation, not fixed

**Phase:** 6 (Calibration UX)
**Decision:** Live calibration's "automatic" pairing means each manual snapshot click auto-pairs and logs itself (removing a previously separate manual "log" button) — not that facial capture itself became continuous/hands-free.
**Reasoning:** True continuous webcam capture without an explicit per-frame user action isn't available through Streamlit's built-in `camera_input` widget; it requires a heavier dependency (`streamlit-webrtc`) with its own real integration risk (browser WebRTC handling, cross-thread frame passing). Given the scope already in flight, this was scoped out rather than attempted under time pressure.
**Consequences:** Calibration data collection is still faster and less manual than the original design (one click instead of a click plus a separate log action), but not the fully hands-free experience initially envisioned. Listed explicitly in the README's Roadmap rather than left unstated.

## D-016: Root-anchoring `.gitignore` patterns for `data/` and `models/`

**Phase:** 9 (Post-audit correction)
**Decision:** Changed `.gitignore`'s `data/` and `models/` entries to `/data/` and `/models/`.
**Reasoning:** Unanchored patterns match a directory of that name anywhere in the repository tree, not just at the root. `emotion_analyzer/emotion_analyzer/models/` — the actual source package containing `architectures.py` and `train.py` — matched the unanchored `models/` pattern and was silently excluded from every commit for this project's entire history. Because gitignored files don't appear in `git status`, this went unnoticed internally through multiple rounds of `git add -A` and review; it was caught by an external audit (EXTERNAL_REVIEW.md F-001), not internally.
**Consequences:** This is the most consequential single bug found in this project to date, not for its complexity (a one-character fix — adding a leading slash) but for what it silently caused: the actual training implementation behind the facial model's headline accuracy number was never verifiable by anyone reading the pushed repository, despite genuinely existing and working the entire time. A README instructing a fresh clone to run `python -m emotion_analyzer.models.train` pointed at code nobody outside the local development machine could actually see.

## D-017: Dataset provenance correction — public model actually trains on all four facial datasets, not FER2013+FER+ alone

**Phase:** 9 (Post-audit correction)
**Decision:** Supersedes D-001. The public facial model's actual training data, as implemented in `loaders.py`/`train.py` and confirmed by `docs/flow.md` and `PROJECT_NARRATIVE.md`, is FER2013 + RAF-DB + AffectNet (confidence-filtered via `relFCs`), with the `contempt` class dropped — not FER2013 + FER+ alone as D-001 originally stated.
**Reasoning:** D-001's licensing-driven restriction (AffectNet/RAF-DB for validation only) was the correct call under an assumption this project was later told explicitly did not apply: that the shipped model needed to avoid any ambiguity for potential commercial/redistribution use. Once the project's actual goal was clarified — a free tool for personal/portfolio use, not a commercial product — that restriction's original justification no longer held, and the training pipeline was built to use all available data without re-examining or updating D-001 to match. The result was three project documents (`flow.md`, `PROJECT_NARRATIVE.md`, `README.md`) correctly describing the actual pipeline while `decisions.md` alone described a policy that was no longer in effect — an internal documentation inconsistency, caught by external audit (EXTERNAL_REVIEW.md F-002) rather than caught internally.
**Consequences:** `README.md`'s licensing language should be read as reflecting a personal/non-commercial-use context, not a "safe for any redistribution" claim — AffectNet, RAF-DB, and DREAMER's original access-request terms still apply to that underlying data regardless of what this project decided to do with it. Anyone forking this project for a different (e.g. commercial) purpose needs to re-evaluate dataset licensing independently; D-001's original reasoning is still correct advice for that different context, which is why it was left in the log rather than deleted.

## D-018: EEG model promotion now uses validation accuracy, not test accuracy

**Phase:** 9 (Post-audit correction)
**Decision:** Added a promotion-only-if-better safeguard to `bci_suite/eeg/train.py` (previously absent — the script always overwrote `models/eeg/eegnet_dreamer.{pt,onnx}` unconditionally), and deliberately used **validation** accuracy as the promotion criterion, not test accuracy.
**Reasoning:** The facial model's existing promotion safeguard (D-007) compares against test accuracy, which external audit correctly flagged as a methodological issue independent of this project (EXTERNAL_REVIEW.md F-015): using the test set as a routine, repeated model-selection criterion erodes its purpose as a held-out estimate. This is more consequential for the EEG model specifically, given its test set is only ~5 participants — repeatedly selecting checkpoints against such a small test group risks overfitting the selection process to that particular held-out group. Validation accuracy is the methodologically correct promotion signal; test accuracy is reported once, after the fact, as an estimate — not used to make decisions.
**Consequences:** The facial model's `train.py` still promotes on test accuracy (D-007) and was not changed to match, since altering an already-working, previously-described pipeline for a different dataset was out of scope for this specific fix — recorded here as a known remaining inconsistency, not silently left unstated. Revisiting D-007 to use validation accuracy instead is a reasonable follow-up, listed in the README roadmap.
