"""
Page 1 — Calibration / Ground Truth.

Live mode: start an EEG stream directly from this page (simulated DREAMER
replay, or a real headset via BrainFlow) — no separate terminal needed.
Every facial snapshot automatically pairs with whatever EEG window is live
at that moment and logs itself. Collect enough pairs to fine-tune a
personalized EEG model, saved under whatever name you choose.

No-hardware mode: falls back to the original DREAMER blind-sample demo.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from emotion_analyzer.inference.predictor import EmotionPredictor
from shared.valence_arousal import facial_label_to_va, va_to_quadrant, QUADRANT_LABELS

st.set_page_config(page_title="Calibration / Ground Truth", page_icon="🎯", layout="wide")
st.title("🎯 Calibration / Ground Truth")

st.info(
    "**Why this page exists:** an EEG model trained on other people's brains "
    "generalizes poorly to any one new person (see the ~48.6% cross-subject "
    "accuracy on the Home page — a known limitation, not a bug). Calibrating "
    "it to *you* needs labeled examples of your own state. Self-reporting "
    "constantly is disruptive — a facial expression is a free, continuous "
    "proxy label instead.",
    icon="💡",
)


@st.cache_resource
def get_facial_predictor():
    return EmotionPredictor()


for key, default in [
    ("eeg_reader", None), ("calibration_pairs", []), ("last_photo_hash", None),
    ("stream_proc", None), ("stream_proc_kind", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

mode = st.radio(
    "EEG source", ["Live stream (headset or replay)", "Sample from DREAMER (no hardware)"],
    horizontal=True,
)

st.divider()

# ============================== LIVE MODE ==============================
if mode == "Live stream (headset or replay)":
    from bci_suite.eeg.stream import EEGStreamReader
    from bci_suite.eeg.stream_launcher import (
        start_replay_stream, start_brainflow_stream, stop_stream, SUPPORTED_BOARDS,
    )

    with st.expander(
        "▶ Start a stream from here"
        + (" — currently running" if st.session_state.stream_proc is not None else ""),
        expanded=st.session_state.stream_proc is None,
    ):
        if st.session_state.stream_proc is None:
            source_kind = st.radio(
                "Source", ["Simulated (DREAMER replay)", "Real headset (via BrainFlow)"],
                horizontal=True, key="source_kind_radio",
            )
            if source_kind == "Simulated (DREAMER replay)":
                if st.button("▶ Start replay stream"):
                    st.session_state.stream_proc = start_replay_stream()
                    st.session_state.stream_proc_kind = "replay"
                    st.info("Starting — loading DREAMER.mat takes a few seconds. "
                             "Click 'Connect to EEG stream' below once it's up.")
            else:
                board_label = st.selectbox("Headset", list(SUPPORTED_BOARDS.keys()))
                serial_port = ""
                if SUPPORTED_BOARDS[board_label]["needs_serial_port"]:
                    serial_port = st.text_input(
                        "Serial port", placeholder="e.g. /dev/cu.usbserial-XXXX"
                    )
                st.caption(
                    "Requires `pip install brainflow`. Note: some ecosystems "
                    "(e.g. Emotiv's official Cortex API) need their own vendor "
                    "software running regardless — not something any library "
                    "here can bridge around."
                )
                if st.button("▶ Start headset stream"):
                    st.session_state.stream_proc = start_brainflow_stream(board_label, serial_port)
                    st.session_state.stream_proc_kind = "brainflow"
                    st.info("Starting — click 'Connect to EEG stream' below once connected.")
        else:
            still_running = st.session_state.stream_proc.poll() is None
            st.success(
                f"{'Running' if still_running else 'Exited'} — "
                f"{st.session_state.stream_proc_kind} (PID {st.session_state.stream_proc.pid})"
            )
            if st.button("⏹ Stop stream"):
                stop_stream(st.session_state.stream_proc)
                st.session_state.stream_proc = None
                st.session_state.eeg_reader = None
                st.rerun()

    conn_col, k_col = st.columns([2, 1])

    with conn_col:
        if st.session_state.eeg_reader is None:
            if st.button("🔌 Connect to EEG stream"):
                reader = EEGStreamReader()
                status = reader.connect()
                if not status.get("connected"):
                    st.error(status["error"])
                else:
                    st.session_state.eeg_reader = reader
                    if status["warning"]:
                        st.warning(status["warning"])
                    else:
                        st.success(
                            f"Connected to **{status['stream_name']}** "
                            f"({status.get('note', 'channels matched')})"
                        )
        else:
            st.success(f"Connected to **{st.session_state.eeg_reader.stream_name}**")
            if st.button("Disconnect"):
                st.session_state.eeg_reader = None
                st.session_state.calibration_pairs = []
                st.rerun()

    with k_col:
        k_target = st.number_input(
            "Target number of pairs (k)", min_value=10, max_value=1000, value=80, step=10
        )
        st.caption(
            "Rough guidance: aim for **at least ~20 examples per emotion "
            "quadrant** (k≈80 for 4 quadrants) as a starting point — a "
            "heuristic from few-shot personalization practice, not a "
            "guarantee. Cover a genuine variety of states while collecting, "
            "not just one or two repeated."
        )

    if st.session_state.eeg_reader is not None:
        reader = st.session_state.eeg_reader
        reader.pull_samples(timeout=0.1)

        st.subheader("📷 Take a snapshot to log a calibration pair")
        img_file = st.camera_input("Capture your expression", key="live_calib_cam")

        if img_file is not None:
            photo_hash = hashlib.md5(img_file.getvalue()).hexdigest()
            is_new_photo = photo_hash != st.session_state.last_photo_hash

            file_bytes = np.frombuffer(img_file.getvalue(), dtype=np.uint8)
            import cv2
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            predictor = get_facial_predictor()
            predictions = predictor.predict_image(bgr)

            if predictions:
                facial_label = predictions[0]["label"]
                va = facial_label_to_va(facial_label)
                quadrant = va_to_quadrant(*va)
                st.success(f"Detected: **{facial_label.capitalize()}** → quadrant **{quadrant}**")

                window = reader.get_latest_window()
                if window is None:
                    st.warning("No fresh EEG window available yet — wait a moment and retake the photo.")
                elif is_new_photo:
                    st.session_state.calibration_pairs.append(
                        {"window": window, "quadrant": quadrant, "facial_label": facial_label}
                    )
                    st.session_state.last_photo_hash = photo_hash
                    st.toast(f"Logged pair #{len(st.session_state.calibration_pairs)} automatically ✓")
            else:
                st.warning("No face detected in the snapshot.")

        n_collected = len(st.session_state.calibration_pairs)
        st.progress(min(n_collected / k_target, 1.0), text=f"{n_collected} / {k_target} pairs collected")

        if st.session_state.calibration_pairs:
            with st.expander(f"View collected pairs ({n_collected})"):
                st.dataframe(
                    [{"Facial label": p["facial_label"].capitalize(), "Quadrant": p["quadrant"]}
                     for p in st.session_state.calibration_pairs],
                    use_container_width=True,
                )

        if n_collected >= k_target:
            st.success(f"Target reached — {n_collected} pairs collected. Ready to train.")
            model_name = st.text_input("Name this personalized model", value="my_model")
            if st.button("🚀 Train personalized model"):
                from bci_suite.eeg.personalize import finetune_from_calibration

                with st.spinner("Fine-tuning on your calibration data..."):
                    windows = [p["window"] for p in st.session_state.calibration_pairs]
                    quadrants = [p["quadrant"] for p in st.session_state.calibration_pairs]
                    metadata = finetune_from_calibration(windows, quadrants, model_name)

                st.success(
                    f"Saved **{metadata['name']}** — trained on {metadata['n_examples']} examples, "
                    f"{metadata['final_train_acc'] * 100:.0f}% final training accuracy. "
                    "Select it on the Fusion Demo page to use it."
                )
        elif n_collected > 0:
            st.caption(f"Need {k_target - n_collected} more pairs before training unlocks.")

# ============================== NO-HARDWARE MODE ==============================
else:
    @st.cache_resource
    def get_eeg_classifier():
        try:
            from bci_suite.eeg.classifier import EEGEmotionClassifier
            return EEGEmotionClassifier()
        except Exception:
            return None

    @st.cache_resource
    def get_sample_eeg_windows(n_per_quadrant: int = 5):
        from bci_suite.eeg.loaders import load_dreamer_windows
        from shared.valence_arousal import IDX_TO_QUADRANT

        X, y, groups = load_dreamer_windows()
        rng = np.random.RandomState(42)
        samples = []
        for label_idx, quadrant in IDX_TO_QUADRANT.items():
            idxs = np.where(y == label_idx)[0]
            chosen = rng.choice(idxs, size=min(n_per_quadrant, len(idxs)), replace=False)
            for i in chosen:
                samples.append(
                    {"window": X[i, 0], "true_quadrant": quadrant, "participant": int(groups[i])}
                )
        return samples

    col1, col2 = st.columns(2, gap="large")

    facial_result = None
    with col1:
        with st.container(border=True):
            st.subheader("📷 Facial channel (the proxy label)")
            img_file = st.camera_input("Capture your expression", key="calib_cam_nohw")
            if img_file is not None:
                import cv2
                file_bytes = np.frombuffer(img_file.getvalue(), dtype=np.uint8)
                bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                predictor = get_facial_predictor()
                predictions = predictor.predict_image(bgr)
                if predictions:
                    facial_result = predictions[0]
                    st.success(
                        f"**{facial_result['label'].capitalize()}** "
                        f"— {facial_result['confidence'] * 100:.0f}% confidence"
                    )
                else:
                    st.warning("No face detected in the snapshot.")
            else:
                st.info("Take a photo above — this becomes the calibration label.")

    with col2:
        with st.container(border=True):
            st.subheader("🧠 EEG channel (blind prediction)")
            eeg_classifier = get_eeg_classifier()
            if eeg_classifier is None:
                st.warning(
                    "No trained EEG model found. Train it first: "
                    "`python -m bci_suite.eeg.train`"
                )
            else:
                samples = get_sample_eeg_windows()
                options = [f"Sample #{i} — participant {s['participant']}" for i, s in enumerate(samples)]
                choice = st.selectbox("Pick an EEG window to classify (label hidden)", options)
                sample = samples[options.index(choice)]

                result = eeg_classifier.predict_window(sample["window"])
                st.success(
                    f"Predicted: **{QUADRANT_LABELS[result['quadrant']]}** "
                    f"({result['confidence'] * 100:.0f}%)"
                )
                with st.expander("Reveal DREAMER's actual self-reported label"):
                    match = result["quadrant"] == sample["true_quadrant"]
                    icon = "✅ matched" if match else "❌ did not match"
                    st.write(f"True label: **{QUADRANT_LABELS[sample['true_quadrant']]}** — {icon}")
