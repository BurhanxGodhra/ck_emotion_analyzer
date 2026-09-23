"""
Replays a DREAMER trial as a live LSL stream, simulating a real EEG headset.
This lets the entire live pipeline (stream.py, and everything built on top
of it) be developed and tested without real hardware — and once real
hardware IS available, it plugs into the exact same code path with zero
changes, since both look identical over LSL. This is the actual point of
building on LSL rather than a custom protocol.

Usage:
    python -m bci_suite.eeg.replay --participant 3 --trial 5
    python -m bci_suite.eeg.replay --random
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.io

from bci_suite.eeg.loaders import DREAMER_MAT_PATH, SAMPLING_RATE_HZ, EEG_CHANNELS


def replay(participant_idx: int, trial_idx: int):
    import pylsl

    mat = scipy.io.loadmat(str(DREAMER_MAT_PATH), squeeze_me=True, struct_as_record=False)
    participants = mat["DREAMER"].Data
    participant = participants[participant_idx]
    trial_eeg = np.asarray(participant.EEG.stimuli[trial_idx])  # (n_samples, 14)
    valence = float(participant.ScoreValence[trial_idx])
    arousal = float(participant.ScoreArousal[trial_idx])

    print(
        f"Replaying participant {participant_idx}, trial {trial_idx} "
        f"(valence={valence}, arousal={arousal}) — {trial_eeg.shape[0]} samples "
        f"at {SAMPLING_RATE_HZ}Hz over LSL..."
    )

    info = pylsl.StreamInfo(
        name="DREAMER-Replay",
        type="EEG",
        channel_count=len(EEG_CHANNELS),
        nominal_srate=SAMPLING_RATE_HZ,
        channel_format="float32",
        source_id="dreamer-replay",
    )
    # Embed channel labels in metadata, matching what a real BrainFlow bridge
    # publishes — this is what stream.py needs to do name-based matching.
    chns = info.desc().append_child("channels")
    for name in EEG_CHANNELS:
        ch = chns.append_child("channel")
        ch.append_child_value("label", name)

    outlet = pylsl.StreamOutlet(info)

    print("Streaming... (Ctrl+C to stop early)")
    interval = 1.0 / SAMPLING_RATE_HZ
    for sample in trial_eeg:
        outlet.push_sample(sample.tolist())
        time.sleep(interval)

    print("Replay finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", type=int, default=0, help="0-22")
    parser.add_argument("--trial", type=int, default=0, help="0-17")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    if args.random:
        import random
        p, t = random.randint(0, 22), random.randint(0, 17)
        print(f"Random pick: participant {p}, trial {t}")
        replay(p, t)
    else:
        replay(args.participant, args.trial)
