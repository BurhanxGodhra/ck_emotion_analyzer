"""
Quick standalone test for the in-app stream launcher. Run with:
    python3 test_stream_launch.py
"""

import time

from bci_suite.eeg.stream_launcher import start_replay_stream, stop_stream
from bci_suite.eeg.stream import EEGStreamReader

proc = start_replay_stream()
print("Waiting for replay to start (loading DREAMER.mat takes a few seconds)...")
time.sleep(8)

print("Subprocess still running:", proc.poll() is None)

r = EEGStreamReader()
status = r.connect()
print("Connect status:", status)

stop_stream(proc)
out, _ = proc.communicate(timeout=2)
print("--- replay.py output ---")
print(out)
