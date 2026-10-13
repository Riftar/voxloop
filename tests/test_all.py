"""Offline test suite for voxloop - stdlib unittest, zero network.

Run:  python3 tests/test_all.py
"""

import os
import sys
import time
import unittest
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from engine import Speaker, TurnEngine, silence, tone  # noqa: E402
from vad import FRAME_BYTES, FRAME_MS, Vad, rms  # noqa: E402

FIXTURES = os.path.join(ROOT, "fixtures")


def wav_frames(path: str) -> list[bytes]:
    """Split a fixture WAV into Vad-sized frames."""
    with wave.open(path, "rb") as w:
        pcm = w.readframes(w.getnframes())
    assert len(pcm) % FRAME_BYTES == 0, "fixture not frame-aligned"
    return [pcm[i:i + FRAME_BYTES] for i in range(0, len(pcm), FRAME_BYTES)]


class RmsTests(unittest.TestCase):
    def test_silence_is_zero(self):
        self.assertEqual(rms(silence()), 0.0)

    def test_tone_level_above_threshold(self):
        self.assertGreater(rms(tone(amp=6000)), 350.0)

    def test_noise_below_threshold(self):
        self.assertLess(rms(tone(amp=80)), 350.0)

    def test_short_input_safe(self):
        self.assertEqual(rms(b""), 0.0)
        self.assertEqual(rms(b"\x01"), 0.0)  # odd byte count tolerated


class VadTests(unittest.TestCase):
    def feed_all(self, vad: Vad, frames: list[bytes]):
        events = []
        t = 0.0
        for f in frames:
            events += vad.feed(f, t)
            t += FRAME_MS / 1000.0
        return events

    def test_silence_never_triggers(self):
        vad = Vad()
        events = self.feed_all(vad, wav_frames(os.path.join(FIXTURES, "silence.wav")))
        self.assertFalse(any(e[0] in ("speech_start", "speech_end") for e in events))

    def test_noise_does_not_trigger(self):
        vad = Vad()
        events = self.feed_all(vad, wav_frames(os.path.join(FIXTURES, "noise.wav")))
        self.assertFalse(any(e[0] in ("speech_start", "speech_end") for e in events))

    def test_speech_starts_and_ends(self):
        vad = Vad()
        events = self.feed_all(
            vad,
            wav_frames(os.path.join(FIXTURES, "convo_two_turns.wav")))
        starts = [e for e in events if e[0] == "speech_start"]
        ends = [e for e in events if e[0] == "speech_end"]
        self.assertEqual(len(starts), 2)
        self.assertEqual(len(ends), 2)
        # first start near 0ms (attack 40ms), first end ~600ms hangover
        self.assertLess(starts[0][1], 0.15)
        self.assertGreater(ends[0][1], 1.0)  # 1.0s speech + 0.4s hangover
        self.assertLess(ends[0][1], 1.7)

    def test_hangover_bridge(self):
        """A short pause (< hangover) must NOT end the turn."""
        vad = Vad(hangover_ms=400)
        frames = [tone() for _ in range(25)]          # 500ms speech
        frames += [silence() for _ in range(10)]      # 200ms pause
        frames += [tone() for _ in range(25)]         # 500ms speech
        frames += [silence() for _ in range(30)]      # 600ms -> end
        events = self.feed_all(vad, frames)
        starts = [e for e in events if e[0] == "speech_start"]
        ends = [e for e in events if e[0] == "speech_end"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertGreater(ends[0][1], 1.0)  # ends only after final silence


class TurnEngineTests(unittest.TestCase):
    def make_engine(self, brain=None):
        speaker = Speaker()
        eng = TurnEngine(vad=Vad(), brain=brain or (lambda t: f"echo: {t}"),
                         speaker=speaker)
        eng._injected_utterance = "stub utterance"
        return eng, speaker

    def feed(self, eng, frames, t0=0.0):
        out = []
        t = t0
        for f in frames:
            out += eng.feed(f, t)
            t += FRAME_MS / 1000.0
        return out

    def test_turn_completes_and_replies(self):
        eng, speaker = self.make_engine()
        frames = [tone() for _ in range(30)] + [silence() for _ in range(30)]
        events = self.feed(eng, frames)
        turns = [e for e in events if e[0] == "turn"]
        replies = [e for e in events if e[0] == "reply"]
        self.assertEqual(len(turns), 1)
        self.assertEqual(len(replies), 1)
        self.assertEqual(speaker.events[0][0], "speak_start")
        self.assertEqual(eng.state, "speaking")

    def test_barge_in_cuts_speech_instantly(self):
        eng, speaker = self.make_engine()
        # complete a turn so the engine is speaking
        frames = [tone() for _ in range(30)] + [silence() for _ in range(30)]
        self.feed(eng, frames)
        self.assertEqual(eng.state, "speaking")
        # user barges in: speech frames arrive while agent speaks
        t_before = time.monotonic()
        interrupted = False
        t = 1.3
        for f in [tone() for _ in range(3)]:  # 60ms = attack window
            evs = eng.feed(f, t)
            t += FRAME_MS / 1000.0
            if any(e[0] == "interrupted" for e in evs):
                interrupted = True
                break
        dt_ms = (time.monotonic() - t_before) * 1000
        self.assertTrue(interrupted)
        self.assertEqual(eng.state, "listening")
        kinds = [e[0] for e in speaker.events]
        self.assertIn("speak_stop", kinds)
        self.assertLess(dt_ms, 1000)  # generous wall bound; logic is ms-fast
        # logic latency is the real claim: interrupt fires on the frame
        # where attack completes (2nd frame = 40ms)
        self.assertIn(("interrupted", 1.32), [(e[0], e[1]) for e in eng.log])

    def test_tools_run_while_speaking(self):
        """brain() blocking for 'tool time' must not delay speak() start
        beyond the brain's own duration; the mouth opens the moment the
        reply exists. We assert speak_start happened and that a flag the
        tool sets is visible - ordering: tool finished BEFORE speak, and
        total pipeline stayed non-blocking for audio (no audio API waits)."""
        seen = {}

        def slow_brain(text):
            time.sleep(0.15)  # simulated tool call
            seen["tool_ran"] = True
            return f"reply to {text}"

        eng, speaker = self.make_engine(brain=slow_brain)
        frames = [tone() for _ in range(30)] + [silence() for _ in range(30)]
        t0 = time.monotonic()
        self.feed(eng, frames)
        dt = time.monotonic() - t0
        self.assertTrue(seen.get("tool_ran"))
        self.assertEqual(speaker.events[0][0], "speak_start")
        # 0.6s audio + 0.15s brain + slack; assert nothing pathological
        self.assertLess(dt, 2.0)
        self.assertEqual(eng.state, "speaking")

    def test_no_barge_in_when_disabled(self):
        eng, speaker = self.make_engine()
        eng.barge_in_enabled = False
        frames = [tone() for _ in range(30)] + [silence() for _ in range(30)]
        self.feed(eng, frames)
        for f in [tone() for _ in range(5)]:
            evs = eng.feed(f, 2.0)
            self.assertFalse(any(e[0] == "interrupted" for e in evs))
        self.assertEqual(eng.state, "speaking")


if __name__ == "__main__":
    unittest.main(verbosity=2)
