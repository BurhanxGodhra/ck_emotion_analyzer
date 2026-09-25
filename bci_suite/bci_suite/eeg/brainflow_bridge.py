"""
Bridges a real EEG headset (via BrainFlow) to an LSL outlet, so the rest of
this system sees it exactly like any other LSL stream — same code path as
replay.py's simulation.

Publishes channel LABELS in the LSL stream's metadata (via BrainFlow's own
get_eeg_names), which is what enables stream.py's name-based channel
matching — this is what lets a headset with MORE than 14 channels still
work, as long as those 14 required positions are present among them.

Requires: pip install brainflow
Usage:
    python -m bci_suite.eeg.brainflow_bridge --board-id 38
    python -m bci_suite.eeg.brainflow_bridge --board-id 0 --serial-port /dev/cu.usbserial-XXXX
"""

from __future__ import annotations

import argparse
import time


def run(board_id: int, serial_port: str = ""):
    import pylsl
    from brainflow.board_shim import BoardShim, BrainFlowInputParams

    params = BrainFlowInputParams()
    if serial_port:
        params.serial_port = serial_port

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    channel_names = BoardShim.get_eeg_names(board_id)
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    print(f"Connected — {len(eeg_channels)} EEG channels: {channel_names}, {sampling_rate}Hz")

    info = pylsl.StreamInfo(
        name="BrainFlow-Live", type="EEG", channel_count=len(eeg_channels),
        nominal_srate=sampling_rate, channel_format="float32", source_id="brainflow-live",
    )
    # Embed channel labels in the stream's metadata — this is what
    # stream.py reads to do name-based matching instead of a raw count check.
    chns = info.desc().append_child("channels")
    for name in channel_names:
        ch = chns.append_child("channel")
        ch.append_child_value("label", name)

    outlet = pylsl.StreamOutlet(info)
    print("Streaming to LSL as 'BrainFlow-Live'... (Ctrl+C to stop)")

    try:
        while True:
            # get_board_data() DRAINS the buffer (removes what it returns);
            # get_current_board_data(n) only PEEKS at the latest n samples
            # without removing them. The original version used the latter,
            # which on a continuous polling loop means samples are read
            # multiple times (duplicated) and the buffer grows unbounded
            # between reads — a real bug, not a style preference. Flagged
            # by external audit (EXTERNAL_REVIEW.md F-011).
            data = board.get_board_data()
            if data.shape[1] > 0:
                for i in range(data.shape[1]):
                    sample = [data[ch][i] for ch in eeg_channels]
                    outlet.push_sample(sample)
            time.sleep(1.0 / sampling_rate)
    except KeyboardInterrupt:
        pass
    finally:
        board.stop_stream()
        board.release_session()
        print("Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-id", type=int, required=True)
    parser.add_argument("--serial-port", type=str, default="")
    args = parser.parse_args()
    run(args.board_id, args.serial_port)
