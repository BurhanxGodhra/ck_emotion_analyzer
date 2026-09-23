"""
Loader for DREAMER.mat.

NOTE: written against DREAMER's documented/commonly-used MATLAB structure
(the same structure torcheeg's DREAMERDataset and multiple published
preprocessing scripts parse) but NOT YET TESTED against the real file —
you don't have it yet at time of writing. Expect this to need one round of
live debugging once your access request is approved, the same way
emotion_analyzer's loaders needed fixes against the real downloaded files.
Run the __main__ block below first and paste me the output before trusting
anything downstream of it.

Expected structure (scipy.io.loadmat with squeeze_me=True, struct_as_record=False):
    mat['DREAMER'].Data -> array of 23 participant structs, each with:
        .EEG.stimuli  -> array of 18 trials, each (n_samples, 14) raw EEG during the stimulus
        .EEG.baseline -> array of 18 trials, each (n_samples, 14) pre-stimulus baseline
        .ScoreValence, .ScoreArousal, .ScoreDominance -> arrays of 18 ratings (1-5 scale)

Each 60-ish-second trial is chunked into fixed-length windows (default 4s at
128Hz = 512 samples) to turn 23*18=414 trials into thousands of training
examples — a single trial-level example per class would be far too little
data for EEGNet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from shared.valence_arousal import va_to_quadrant, QUADRANT_TO_IDX

DREAMER_MAT_PATH = Path("data/raw/dreamer/DREAMER.mat")

SAMPLING_RATE_HZ = 128
WINDOW_SECONDS = 4
WINDOW_SAMPLES = SAMPLING_RATE_HZ * WINDOW_SECONDS  # 512
VALENCE_AROUSAL_THRESHOLD = 3.0  # DREAMER ratings are 1-5; midpoint split, matching
                                  # the torcheeg reference usage (transforms.Binary(3.0))

EEG_CHANNELS = [  # standard Emotiv EPOC 14-channel montage, fixed order
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]


def _chunk_into_windows(signal: np.ndarray, window_samples: int) -> list[np.ndarray]:
    """signal: (n_samples, n_channels). Drops a trailing partial window."""
    n_windows = signal.shape[0] // window_samples
    return [
        signal[i * window_samples : (i + 1) * window_samples].T  # -> (n_channels, window_samples)
        for i in range(n_windows)
    ]


def load_dreamer_windows(
    mat_path: Path = DREAMER_MAT_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X: (n_windows, 1, n_channels, window_samples) float32
        y: (n_windows,) int64 — quadrant index
        groups: (n_windows,) int64 — which participant (0-22) each window came
            from. Required for a leakage-free split: windows from the same
            trial are highly correlated (same person, same stimulus), so a
            plain random split can put windows from one trial in both train
            and test, inflating accuracy. Split by participant instead — see
            train.py's use of GroupShuffleSplit.
    """
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    dreamer = mat["DREAMER"]
    participants = dreamer.Data

    X_list, y_list, group_list = [], [], []
    for p_idx, participant in enumerate(participants):
        stimuli = participant.EEG.stimuli
        valence_scores = participant.ScoreValence
        arousal_scores = participant.ScoreArousal

        for trial_idx in range(len(stimuli)):
            trial_eeg = np.asarray(stimuli[trial_idx])  # (n_samples, 14)
            valence = float(valence_scores[trial_idx])
            arousal = float(arousal_scores[trial_idx])

            v_bin = "high" if valence >= VALENCE_AROUSAL_THRESHOLD else "low"
            a_bin = "high" if arousal >= VALENCE_AROUSAL_THRESHOLD else "low"
            quadrant = va_to_quadrant(
                valence=1.0 if v_bin == "high" else -1.0,
                arousal=1.0 if a_bin == "high" else -1.0,
            )
            label_idx = QUADRANT_TO_IDX[quadrant]

            for window in _chunk_into_windows(trial_eeg, WINDOW_SAMPLES):
                X_list.append(window)
                y_list.append(label_idx)
                group_list.append(p_idx)

        print(f"[dreamer] participant {p_idx + 1}/{len(participants)} processed")

    X = np.stack(X_list).astype(np.float32)[:, np.newaxis, :, :]
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(group_list, dtype=np.int64)
    return X, y, groups


if __name__ == "__main__":
    X, y, groups = load_dreamer_windows()
    print(f"\nX shape: {X.shape}  (expected: (N, 1, 14, {WINDOW_SAMPLES}))")
    print(f"y shape: {y.shape}")
    print(f"groups shape: {groups.shape}, {len(np.unique(groups))} unique participants")
    unique, counts = np.unique(y, return_counts=True)
    from shared.valence_arousal import IDX_TO_QUADRANT
    print("Class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {IDX_TO_QUADRANT[u]}: {c}")
