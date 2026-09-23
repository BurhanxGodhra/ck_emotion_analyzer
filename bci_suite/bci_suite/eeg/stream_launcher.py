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
SUPPORTED_BOARDS = {
    "OpenBCI Cyton (8ch, serial)": {"board_id": 0, "needs_serial_port": True},
    "OpenBCI Cyton + Daisy (16ch, serial)": {"board_id": 2, "needs_serial_port": True},
    "OpenBCI Ganglion (4ch, serial)": {"board_id": 1, "needs_serial_port": True},
    "Muse 2 (BLE)": {"board_id": 38, "needs_serial_port": False},
    "Muse S (BLE)": {"board_id": 39, "needs_serial_port": False},
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
