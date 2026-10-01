"""
bci_suite — multimodal (facial + EEG) affective computing demo.
Run with: streamlit run bci_suite/Home.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="BCI Affect Suite", page_icon="🧠", layout="wide")

st.title("🧠 Multimodal Affect Recognition Suite")

st.markdown(
    """
This app pairs two independently-built systems on a shared valence/arousal
vocabulary (see `shared/valence_arousal.py`):

- **Facial emotion analyzer** — a MobileNetV2 model (see the standalone
  `emotion_analyzer` package) classifying 7 discrete emotions from webcam images.
  **67.4% test accuracy.**
- **EEG affect classifier** — an EEGNet model (ported from the
  [mi-bci-pipeline](https://github.com/BurhanxGodhra/mi-bci-pipeline) project)
  trained on the DREAMER dataset, classifying EEG windows into one of 4
  valence/arousal quadrants. **A small, statistically inconclusive signal
  above chance** (29.3% vs. 25% chance, p=0.182 — not significant; see
  `docs/decisions.md` D-019 for the full story, including an earlier,
  overstated single-run figure of 48.6% that didn't hold up under proper
  cross-validation).
"""
)

st.warning(
    "**On its own, a facial-expression classifier has limited value** — "
    "plenty of apps can already detect \"this face looks happy.\" The point "
    "of this suite is that the facial model only becomes useful once it's "
    "*embedded as a component* in a larger system, not run standalone. "
    "Here it plays two specific roles: a cheap, non-disruptive **proxy "
    "label** for calibrating a person-specific EEG model (page 1), and a "
    "**redundant, hard-to-mask-independently channel** that hedges against "
    "either signal being misleading alone (page 2). Judge it by what it "
    "enables, not by what it does in isolation.",
    icon="⚠️",
)

st.info(
    "**On the EEG model's accuracy:** treat it as a demonstrated pipeline, "
    "not a demonstrated working classifier yet. Repeated cross-validation "
    "shows a modest, positive-but-inconclusive signal — not strong enough "
    "to claim this reliably detects affect from EEG. This was found via "
    "external audit and follow-up testing, not caught internally — see "
    "`docs/decisions.md` D-019 for the complete, honest account.",
    icon="📊",
)

st.markdown(
    """
**Pages** (see sidebar):
1. **Calibration / Ground Truth** — an EEG window is classified *blind* (its
   true label is hidden until after prediction). You capture your own facial
   expression as a proxy label, and can log the pair — the same kind of
   record a real system would accumulate to fine-tune the EEG model to one
   specific person.
2. **Multimodal Fusion Demo** — combines both channels into one affect
   estimate. States explicitly why fusion matters: facial expression can be
   consciously suppressed, EEG can't be as easily — agreement between
   channels is corroborating evidence, disagreement is itself informative.
3. **Accessibility Mode** — a facial-only fallback view (large text, high
   contrast) for when EEG is noisy, unavailable, or not the right channel
   for a given user.

---
**Status note:** the EEG classifier requires DREAMER data + a trained model
(`models/eeg/eegnet_dreamer.onnx`) — pages handle a missing model gracefully
and show what's missing rather than crashing.
"""
)
