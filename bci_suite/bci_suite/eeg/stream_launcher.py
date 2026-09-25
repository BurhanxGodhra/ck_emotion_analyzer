"""
Starts and stops EEG-producing processes from within the app itself —
no separate terminal required. Two kinds of source:

1. Simulated (DREAMER replay) — fully within our control, just launches
   replay.py as a background subprocess.
2. Real headset via BrainFlow — BrainFlow (pip install brainflow) supports a
   wide range of consumer/research boards (OpenBCI Cyton/Ganglion, Muse 2/S,
   and others) through ONE unified API, and can push straight to LSL. This
   is why BrainFlow specifically, rather than one-off per-brand SDKs.

Honest limit: some ecosystems (notably Emotiv's official Cortex API) require
their own vendor software running regardless of what we build here — that's
outside anything a Python library can bridge around.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

# A small curated set of common BrainFlow-supported boards. BrainFlow itself
# supports many more (see brainflow.board_shim.BoardIds) — extend this list
# as needed rather than exposing the raw enum, so the UI stays approachable.
#
# "channel_count" and "model_compatible" are stated honestly, not
# optimistically: the EEG model needs all 14 of DREAMER's Emotiv-EPOC-montage
# electrode positions (see EEG_CHANNELS in loaders.py). A board with fewer
# than 14 total channels can NEVER satisfy this regardless of any matching
# logic — Muse 2/S (4 channels) and Ganglion (4 channels) are listed here
# for completeness (BrainFlow does support them for other purposes) but
# flagged incompatible with THIS model. Cyton alone (8 channels) is also
# incompatible. Only Cyton+Daisy (16 channels) has enough raw channels to
# possibly work — "possibly" because BrainFlow/OpenBCI's default channel
# naming for that board is generic (e.g. numbered, not standard 10-20
# labels like "AF3") unless the user has explicitly configured their
# montage to report standard names; name-based matching (stream.py) will
# correctly refuse to proceed if those names aren't present, rather than
# guessing. Flagged by external audit (EXTERNAL_REVIEW.md F-012) — the
# original version listed all boards without this distinction.
SUPPORTED_BOARDS = {
    "OpenBCI Cyton + Daisy (16ch, serial)": {
        "board_id": 2, "needs_serial_port": True, "channel_count": 16,
        "model_compatible": "possible",
    },
    "OpenBCI Cyton (8ch, serial)": {
        "board_id": 0, "needs_serial_port": True, "channel_count": 8,
        "model_compatible": "no",
    },
    "OpenBCI Ganglion (4ch, serial)": {
        "board_id": 1, "needs_serial_port": True, "channel_count": 4,
        "model_compatible": "no",
    },
    "Muse 2 (BLE)": {
        "board_id": 38, "needs_serial_port": False, "channel_count": 4,
        "model_compatible": "no",
    },
    "Muse S (BLE)": {
        "board_id": 39, "needs_serial_port": False, "channel_count": 4,
        "model_compatible": "no",
    },
}


def start_replay_stream(participant: Optional[int] = None, trial: Optional[int] = None) -> subprocess.Popen:
    """Launches replay.py in the background. Returns the process handle —
    keep it (e.g. in st.session_state) so it can be stopped later."""
    cmd = [sys.executable, "-m", "bci_suite.eeg.replay"]
    if participant is not None and trial is not None:
        cmd += ["--participant", str(participant), "--trial", str(trial)]
    else:
        cmd += ["--random"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def start_brainflow_stream(board_label: str, serial_port: str = "") -> subprocess.Popen:
    """Launches brainflow_bridge.py in the background for a real headset."""
    board_info = SUPPORTED_BOARDS[board_label]
    cmd = [
        sys.executable, "-m", "bci_suite.eeg.brainflow_bridge",
        "--board-id", str(board_info["board_id"]),
    ]
    if board_info["needs_serial_port"]:
        cmd += ["--serial-port", serial_port]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def stop_stream(proc: subprocess.Popen):
    if proc is not None and proc.poll() is None:  # still running
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
