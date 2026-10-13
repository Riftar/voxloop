"""Real-speech tests for voxloop, using the TTS fixtures in fixtures/speech/.

Unlike test_all.py (synthetic tones), these run the VAD/turn engine against
actual speech audio with known transcripts. Requires the fixtures generated
by ingest (see manifest.json); all assertions are offline and deterministic.

Run:  python3 tests/test_speech.py
"""

import json
import os
import sys
import unittest
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from engine import Speaker, TurnEngine  # noqa: E402
from vad import FRAME_BYTES, FRAME_MS, Vad  # noqa: E402

SPEECH_DIR = os.path.join(ROOT, "fixtures", "speech")
MANIFEST = os.path.join(SPEECH_DIR, "manifest.json")

TRAILING_SILENCE_S = 0.7  # appended after each file: the room going quiet


def load_frames(name: str) -> list[bytes]:
    with wave.open(os.path.join(SPEECH_DIR, name), "rb") as w:
        pcm = w.readframes(w.getnframes())
    return [pcm[i:i + FRAME_BYTES]
            for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES)]


def silence_frames(seconds: float) -> list[bytes]:
    n = int(seconds * 1000 // FRAME_MS)
    return [bytes(FRAME_BYTES) for _ in range(n)]


def run_vad(vad: Vad, frames: list[bytes]) -> dict:
    starts, ends, t = [], [], 0.0
    for f in frames:
        for ev in vad.feed(f, t):
            if ev[0] == "speech_start":
                starts.append(ev[1])
            elif ev[0] == "speech_end":
                ends.append(ev[1])
        t += FRAME_MS / 1000.0
    return {"starts": starts, "ends": ends, "n_turns": len(ends)}


@unittest.skipUnless(os.path.isdir(SPEECH_DIR), "speech fixtures not present")
class SpeechFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(MANIFEST) as f:
            cls.manifest = json.load(f)["files"]

    def turn_count(self, name: str, vad: Vad | None = None) -> int:
        meta = self.manifest[name]
        frames = load_frames(name) + silence_frames(TRAILING_SILENCE_S)
        result = run_vad(vad or Vad(), frames)
        # every utterance must fully close: starts == ends == turns
        self.assertEqual(len(result["starts"]), result["n_turns"])
        return result["n_turns"]

    def test_01_every_utterance_takes_exactly_one_turn(self):
        single = [n for n, m in self.manifest.items()
                  if m["expected_segments"] == 1]
        for name in single:
            self.assertEqual(self.turn_count(name), 1,
                             f"{name}: {self.manifest[name]}")

    def test_02_short_utterance_still_detected(self):
        # "Yes." — 0.96s: attack gate must catch it, turn must complete
        self.assertEqual(self.turn_count("utt_03_en_short.wav"), 1)

    def test_03_pause_default_hangover_splits(self):
        # 770ms mid-sentence pause > 400ms default hangover -> 2 turns.
        # Documents current behavior; the adaptive-hangover enhancement
        # (voxloop Checkpoint 2) targets exactly this case.
        self.assertEqual(self.turn_count("utt_04_en_pause.wav"), 2)

    def test_04_pause_long_hangover_bridges(self):
        # raising the hangover to 1000ms must merge the pause into one turn
        self.assertEqual(
            self.turn_count("utt_04_en_pause.wav", Vad(hangover_ms=1000)), 1)

    def test_05_barge_in_cuts_agent_speech(self):
        # engine is speaking; user says "Wait, hold on."
        eng = TurnEngine(vad=Vad(), brain=lambda t: "long reply", speaker=Speaker())
        eng._injected_utterance = "wait hold on"
        # force the speaking state
        eng.state = "speaking"
        interrupted = False
        t = 0.0
        for f in load_frames("utt_08_en_barge.wav"):
            evs = eng.feed(f, t)
            t += FRAME_MS / 1000.0
            if any(e[0] == "interrupted" for e in evs):
                interrupted = True
                break
        self.assertTrue(interrupted)
        self.assertEqual(eng.state, "listening")

    def test_06_manifest_ground_truth_complete(self):
        for name, meta in self.manifest.items():
            self.assertIn("transcript", meta, name)
            self.assertIn("ground_truth", meta, name)
            self.assertGreater(len(meta["transcript"]), 0, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
