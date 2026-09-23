"""
Page 2 — Multimodal Fusion Demo.

Why fusion matters: facial expression can be consciously suppressed; EEG
can't be as easily. Agreement between channels is corroborating evidence,
disagreement is itself informative.

Supports selecting between the generic DREAMER-trained model and any
personalized model you've trained on the Calibration page, and starting a
live EEG stream (simulated or real headset via BrainFlow) directly from
this page — no separate terminal needed — or falling back to a DREAMER
blind-sample for anyone without hardware.
"""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from emotion_analyzer.inference.predictor import EmotionPredictor
from shared.valence_arousal import facial_label_to_va, va_to_quadrant, QUADRANT_LABELS

QUADRANT_TO_VA = {
    "HVHA": (0.6, 0.6), "HVLA": (0.6, -0.6), "LVHA": (-0.6, 0.6), "LVLA": (-0.6, -0.6),
}

st.set_page_config(page_title="Multimodal Fusion Demo", page_icon="🔀", layout="wide")
st.title("🔀 Multimodal Fusion Demo")

st.info(
    "**Why fuse two channels instead of trusting one?** Facial expression "
    "can be consciously suppressed. EEG reflects internal state more "
    "directly and is far harder to mask. Agreement between channels is "
    "corroborating evidence; disagreement is itself informative.",
    icon="💡",
)


@st.cache_resource
def get_facial_predictor():
    return EmotionPredictor()


def load_eeg_classifier(model_choice: str):
    from bci_suite.eeg.classifier import EEGEmotionClassifier
    if model_choice == "Generic (DREAMER-trained, 48.6% cross-subject)":
        return EEGEmotionClassifier()
    from bci_suite.eeg.personalize import PERSONALIZED_DIR
    return EEGEmotionClassifier(
        model_path=PERSONALIZED_DIR / f"{model_choice}.onnx",
        labels_path=PERSONALIZED_DIR / f"{model_choice}_labels.json",
    )


def get_model_options() -> list[str]:
    options = ["Generic (DREAMER-trained, 48.6% cross-subject)"]
    try:
        from bci_suite.eeg.personalize import list_personalized_models
        options += [m["name"] for m in list_personalized_models()]
    except Exception:
        pass
    return options


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
            samples.append({"window": X[i, 0], "true_quadrant": quadrant, "participant": int(groups[i])})
    return samples


def plot_circumplex(points: dict[str, tuple[float, float]]):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    ax.set_xlabel("Valence"); ax.set_ylabel("Arousal")
    colors = {"Facial": "#4F8DFF", "EEG": "#B07CFF", "Fused": "#37D399"}
    for name, (v, a) in points.items():
        ax.scatter(v, a, s=140, label=name, color=colors.get(name, "#5A6270"), zorder=3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Valence/Arousal Circumplex", fontsize=11)
    st.pyplot(fig, use_container_width=True)


for key, default in [("fusion_eeg_reader", None), ("fusion_stream_proc", None), ("fusion_stream_proc_kind", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

model_choice = st.selectbox("EEG model to use", get_model_options())
eeg_source = st.radio(
    "EEG source", ["Live stream (headset or replay)", "Sample from DREAMER (no hardware)"],
    horizontal=True,
)

st.divider()

col1, col2 = st.columns(2, gap="large")

with col1:
    with st.container(border=True):
        st.subheader("📷 Facial channel")
        img_file = st.camera_input("Capture your expression", key="fusion_cam")

eeg_quadrant, sample = None, None
with col2:
    with st.container(border=True):
        st.subheader("🧠 EEG channel")
        try:
            eeg_classifier = load_eeg_classifier(model_choice)
        except Exception as e:
            eeg_classifier = None
            st.warning(f"Couldn't load this model: {e}")

        if eeg_classifier is not None:
            if eeg_source == "Live stream (headset or replay)":
                from bci_suite.eeg.stream import EEGStreamReader
                from bci_suite.eeg.stream_launcher import (
                    start_replay_stream, start_brainflow_stream, stop_stream, SUPPORTED_BOARDS,
                )

                with st.expander(
                    "▶ Start a stream from here"
                    + (" — running" if st.session_state.fusion_stream_proc is not None else ""),
                    expanded=st.session_state.fusion_stream_proc is None
                    and st.session_state.fusion_eeg_reader is None,
                ):
                    if st.session_state.fusion_stream_proc is None:
                        source_kind = st.radio(
                            "Source", ["Simulated (DREAMER replay)", "Real headset (via BrainFlow)"],
                            horizontal=True, key="fusion_source_kind",
                        )
                        if source_kind == "Simulated (DREAMER replay)":
                            if st.button("▶ Start replay stream", key="fusion_start_replay"):
                                st.session_state.fusion_stream_proc = start_replay_stream()
                                st.session_state.fusion_stream_proc_kind = "replay"
                                st.info("Starting — takes a few seconds, then click 'Connect' below.")
                        else:
                            board_label = st.selectbox(
                                "Headset", list(SUPPORTED_BOARDS.keys()), key="fusion_board_select"
                            )
                            serial_port = ""
                            if SUPPORTED_BOARDS[board_label]["needs_serial_port"]:
                                serial_port = st.text_input(
                                    "Serial port", placeholder="e.g. /dev/cu.usbserial-XXXX",
                                    key="fusion_serial_port",
                                )
                            if st.button("▶ Start headset stream", key="fusion_start_brainflow"):
                                st.session_state.fusion_stream_proc = start_brainflow_stream(
                                    board_label, serial_port
                                )
                                st.session_state.fusion_stream_proc_kind = "brainflow"
                                st.info("Starting — click 'Connect' below once connected.")
                    else:
                        still_running = st.session_state.fusion_stream_proc.poll() is None
                        st.success(
                            f"{'Running' if still_running else 'Exited'} — "
                            f"{st.session_state.fusion_stream_proc_kind}"
                        )
                        if st.button("⏹ Stop stream", key="fusion_stop_stream"):
                            stop_stream(st.session_state.fusion_stream_proc)
                            st.session_state.fusion_stream_proc = None
                            st.session_state.fusion_eeg_reader = None
                            st.rerun()

                if st.session_state.fusion_eeg_reader is None:
                    if st.button("🔌 Connect to EEG stream", key="fusion_connect"):
                        reader = EEGStreamReader()
                        status = reader.connect()
                        if status.get("connected"):
                            st.session_state.fusion_eeg_reader = reader
                            if status["warning"]:
                                st.warning(status["warning"])
                        else:
                            st.error(status["error"])
                else:
                    reader = st.session_state.fusion_eeg_reader
                    reader.pull_samples(timeout=0.1)
                    window = reader.get_latest_window()
                    if window is None:
                        st.info("Waiting for enough live samples to fill a window...")
                    else:
                        result = eeg_classifier.predict_window(window)
                        eeg_quadrant = result["quadrant"]
                        st.success(f"Predicted: **{QUADRANT_LABELS[eeg_quadrant]}** ({result['confidence'] * 100:.0f}%)")
            else:
                samples = get_sample_eeg_windows()
                options = [f"Sample #{i} — participant {s['participant']}" for i, s in enumerate(samples)]
                choice = st.selectbox("Pick an EEG window (label hidden)", options, key="fusion_eeg_sample")
                sample = samples[options.index(choice)]
                eeg_result = eeg_classifier.predict_window(sample["window"])
                eeg_quadrant = eeg_result["quadrant"]
                st.success(f"Predicted: **{QUADRANT_LABELS[eeg_quadrant]}** ({eeg_result['confidence'] * 100:.0f}%)")
                with st.expander("Reveal DREAMER's actual self-reported label"):
                    match = eeg_quadrant == sample["true_quadrant"]
                    icon = "✅ matched" if match else "❌ did not match"
                    st.write(f"True label: **{QUADRANT_LABELS[sample['true_quadrant']]}** — {icon}")

st.divider()

if img_file is not None:
    import cv2
    file_bytes = np.frombuffer(img_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    predictor = get_facial_predictor()
    predictions = predictor.predict_image(bgr)

    if not predictions:
        st.warning("No face detected — capture a photo with a visible face to see the fusion result.")
    else:
        facial_label = predictions[0]["label"]
        facial_va = facial_label_to_va(facial_label)
        facial_quadrant = va_to_quadrant(*facial_va)

        points = {"Facial": facial_va}
        if eeg_quadrant is not None:
            eeg_va = QUADRANT_TO_VA[eeg_quadrant]
            points["EEG"] = eeg_va
            fused_va = ((facial_va[0] + eeg_va[0]) / 2, (facial_va[1] + eeg_va[1]) / 2)
            if facial_quadrant == eeg_quadrant:
                st.success(f"✅ Channels agree — both indicate **{QUADRANT_LABELS[eeg_quadrant]}**.")
            else:
                st.warning(
                    f"⚠️ Channels disagree — facial suggests **{QUADRANT_LABELS[facial_quadrant]}**, "
                    f"EEG suggests **{QUADRANT_LABELS[eeg_quadrant]}**."
                )
        else:
            fused_va = facial_va

        points["Fused"] = fused_va
        result_col1, result_col2 = st.columns([2, 1], gap="large")
        with result_col1:
            plot_circumplex(points)
        with result_col2:
            st.metric("Fused valence", f"{fused_va[0]:+.2f}")
            st.metric("Fused arousal", f"{fused_va[1]:+.2f}")
            st.caption(f"Facial channel detected: **{facial_label.capitalize()}**")
else:
    st.info("Take a photo above (facial channel) to see the fusion result here.")
