"""
Live EEG stream reader over LSL. Works identically whether the stream comes
from a real headset (via brainflow_bridge.py) or replay.py's simulation.

Channel matching is done BY NAME, not by raw channel count — if the stream
publishes channel labels (both replay.py and brainflow_bridge.py do this)
and all 14 of the model's required electrode positions (see EEG_CHANNELS)
are present among them, we select and reorder just those 14, regardless of
how many total channels the device has. A 32-channel research cap can work
fine this way; a 4-channel consumer headset that simply lacks several of
these positions cannot — that's a hardware fact, not something matching
logic can work around.

If a stream publishes NO channel labels at all (some minimal/legacy LSL
outlets don't), we fall back to a raw count check and warn that channel
order is being assumed rather than verified.

Requires: pip install pylsl
"""

from __future__ import annotations

import collections
from typing import Optional

import numpy as np

from bci_suite.eeg.loaders import EEG_CHANNELS, WINDOW_SAMPLES


def _read_channel_labels(stream_info) -> list[str]:
    """Extracts channel labels from an LSL StreamInfo's metadata, if present."""
    labels = []
    ch = stream_info.desc().child("channels").child("channel")
    while not ch.empty():
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return labels


class EEGStreamReader:
    def __init__(
        self,
        expected_channels: list[str] = EEG_CHANNELS,
        window_samples: int = WINDOW_SAMPLES,
    ):
        self.expected_channels = expected_channels
        self.window_samples = window_samples
        self.inlet = None
        self.buffer: collections.deque = collections.deque(maxlen=window_samples * 4)
        self.stream_name: Optional[str] = None
        self.channel_select_indices: Optional[list[int]] = None  # maps stream's
        # raw channel order -> the 14 needed, in our required order

    def list_available_streams(self) -> list[dict]:
        import pylsl
        streams = pylsl.resolve_byprop("type", "EEG", timeout=3.0)
        return [
            {"name": s.name(), "channel_count": s.channel_count(), "srate": s.nominal_srate()}
            for s in streams
        ]

    def connect(self, stream_name: Optional[str] = None) -> dict:
        import pylsl
        streams = pylsl.resolve_byprop("type", "EEG", timeout=5.0)
        if not streams:
            return {
                "connected": False,
                "error": "No LSL EEG streams found. Start one first (replay or a real headset).",
            }

        chosen = streams[0]
        if stream_name:
            matches = [s for s in streams if s.name() == stream_name]
            if matches:
                chosen = matches[0]

        self.inlet = pylsl.StreamInlet(chosen)
        # resolve_byprop() only returns a LIGHTWEIGHT StreamInfo (name, type,
        # channel count, sample rate) — custom metadata like channel labels
        # isn't included until you open an inlet and fetch the full info via
        # inlet.info(), which does a round-trip to the source for the
        # complete description. Reading labels off `chosen` directly (the
        # lightweight object) always comes back empty — this is the actual
        # bug behind the "no channel labels" fallback showing up even when
        # replay.py/brainflow_bridge.py correctly published them.
        full_info = self.inlet.info()
        self.stream_name = full_info.name()
        self.buffer.clear()

        raw_channel_count = full_info.channel_count()
        labels = _read_channel_labels(full_info)

        if labels:
            labels_lower = [l.lower() for l in labels]
            missing = [c for c in self.expected_channels if c.lower() not in labels_lower]
            if missing:
                return {
                    "connected": True, "stream_name": self.stream_name,
                    "channel_count": raw_channel_count, "channel_match": False,
                    "warning": (
                        f"This stream is missing required electrode position(s): "
                        f"{', '.join(missing)}. It has: {', '.join(labels)}. "
                        f"This headset's montage cannot supply what the model needs — "
                        f"no channel-matching logic can work around missing physical "
                        f"electrodes."
                    ),
                }
            # Build the index mapping: for each of our required channels, find
            # its position in the stream's actual channel order.
            self.channel_select_indices = [labels_lower.index(c.lower()) for c in self.expected_channels]
            return {
                "connected": True, "stream_name": self.stream_name,
                "channel_count": raw_channel_count, "channel_match": True,
                "warning": None,
                "note": f"Matched by name — using {len(self.expected_channels)} of "
                        f"{raw_channel_count} available channels.",
            }
        else:
            # No labels published at all — fall back to a raw count check,
            # and be explicit that channel order is an assumption, not verified.
            if raw_channel_count == len(self.expected_channels):
                self.channel_select_indices = list(range(len(self.expected_channels)))
                return {
                    "connected": True, "stream_name": self.stream_name,
                    "channel_count": raw_channel_count, "channel_match": True,
                    "warning": "This stream publishes no channel labels — channel "
                               "count matches, but order is ASSUMED, not verified.",
                }
            return {
                "connected": True, "stream_name": self.stream_name,
                "channel_count": raw_channel_count, "channel_match": False,
                "warning": f"This stream publishes no channel labels and has "
                           f"{raw_channel_count} channels, not the {len(self.expected_channels)} "
                           f"required — cannot use it with this model.",
            }

    def pull_samples(self, timeout: float = 0.0) -> int:
        if self.inlet is None:
            return 0
        chunk, _timestamps = self.inlet.pull_chunk(timeout=timeout)
        for sample in chunk:
            self.buffer.append(sample)
        return len(chunk)

    def get_latest_window(self) -> Optional[np.ndarray]:
        if self.channel_select_indices is None:
            return None
        if len(self.buffer) < self.window_samples:
            return None
        recent = list(self.buffer)[-self.window_samples:]
        arr = np.array(recent, dtype=np.float32)  # (window_samples, raw_channel_count)
        selected = arr[:, self.channel_select_indices]  # -> (window_samples, 14), correctly ordered
        return selected.T  # -> (14, window_samples)
