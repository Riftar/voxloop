"""The turn engine: interruption handling + tools-while-talking.

The engine is audio-source-agnostic. Frames come from anywhere (mic,
file, another process); the engine decides *when* turns happen and *what
gets cut* when the user barges in. Real STT/TTS are adapters behind two
interfaces; the offline test rig fakes both deterministically.
"""

from __future__ import annotations

import time
import array
import math
from typing import Callable, Optional

from vad import FRAME_BYTES, FRAME_MS, Vad


class Speaker:
    """Playback sink abstraction. The fake one records timestamps.

    A real adapter pushes PCM to a device or TTS subprocess; the engine
    only requires: speak(text) begins playback, stop() cuts it.
    """

    def __init__(self):
        self.events: list[tuple] = []

    def speak(self, text: str) -> None:
        self.events.append(("speak_start", round(time.monotonic(), 4), text))

    def stop(self) -> None:
        self.events.append(("speak_stop", round(time.monotonic(), 4)))


class TurnEngine:
    """Wires VAD, a speaker, and a brain callable into a turn loop.

    brain(text) -> reply: any callable. In production it is
    STT -> LLM/tools -> TTS-text; in tests it is a stub. Tools run
    inside brain(); because speak() is non-blocking, the mouth keeps
    moving while tools work.
    """

    def __init__(self, vad: Optional[Vad] = None,
                 brain: Optional[Callable[[str], str]] = None,
                 speaker: Optional[Speaker] = None,
                 barge_in_enabled: bool = True):
        self.vad = vad or Vad()
        self.brain = brain or (lambda text: f"echo: {text}")
        self.speaker = speaker or Speaker()
        self.barge_in_enabled = barge_in_enabled
        self.state = "idle"  # idle | listening | thinking | speaking
        self.log: list[tuple] = []

    # ---- engine API -------------------------------------------------------

    def feed(self, pcm: bytes, t: float | None = None) -> list[tuple]:
        """Feed one audio frame while the agent is speaking or listening.

        Barge-in: any speech_start while state == speaking cuts playback
        immediately and returns to listening.
        """
        t = t if t is not None else time.monotonic()
        events = self.vad.feed(pcm, t)
        out: list[tuple] = []
        for ev in events:
            kind = ev[0]
            if kind == "speech_start" and self.barge_in_enabled \
                    and self.state == "speaking":
                self.speaker.stop()          # <-- the sub-1s guarantee:
                self.state = "listening"     #     cut first, ask later
                out.append(("interrupted", t))
                self.log.append(("interrupted", round(t, 4)))
            elif kind == "speech_start":
                if self.state != "speaking":
                    self.state = "listening"
                out.append(ev)
            elif kind == "speech_end":
                if self.state == "listening":
                    utterance = self._transcribe_window()
                    self.state = "thinking"
                    out.append(("turn", t, utterance))
                    # brain may run tools; speak() is non-blocking, so the
                    # pipeline never waits on tools to free the mouth.
                    reply = self.brain(utterance)
                    self.state = "speaking"
                    self.speaker.speak(reply)
                    out.append(("reply", t, reply))
                else:
                    out.append(ev)
            else:
                out.append(ev)
        return out

    def _transcribe_window(self) -> str:
        """Real builds: feed the buffered window to whisper.cpp. Offline
        test rig: the harness injects the utterance directly."""
        return getattr(self, "_injected_utterance", "(untranscribed)")


# --------------------------------------------------------------------------
# Offline test rig: synthetic PCM
# --------------------------------------------------------------------------

def tone(freq: float = 220.0, ms: int = FRAME_MS, amp: int = 6000,
         rate: int = 16000) -> bytes:
    """A synthetic 'speech' frame: voiced-frequency sine burst."""
    n = rate * ms // 1000
    out = []
    for i in range(n):
        out.append(int(amp * math.sin(2 * math.pi * freq * i / rate)))
    return array.array("h", out).tobytes()


def silence(ms: int = FRAME_MS) -> bytes:
    return bytes(FRAME_BYTES) if ms == FRAME_MS else bytes(
        FRAME_BYTES * ms // FRAME_MS)
