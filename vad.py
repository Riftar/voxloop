"""voxloop - a real-time voice turn engine: VAD, barge-in interruption,
and tool-calls-while-talking, in pure stdlib Python.

The design target is sub-1s interruption handling:

  mic frames -> [VAD] --speech_start--> cut playback instantly (barge-in)
                       --speech_end---> turn taken: transcribe -> think
                                        (tools may run while speaking)
                                        -> synthesize reply

Nothing here requires a network, a model, or a third-party package. Real
STT/TTS plug in later as CLI subprocess adapters (whisper.cpp / piper);
the engine and its tests run offline on synthetic PCM.
"""

from __future__ import annotations

import array
import math

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000      # 320 samples
FRAME_BYTES = FRAME_SAMPLES * 2                     # 16-bit mono


def rms(pcm: bytes) -> float:
    """Root-mean-square level of 16-bit little-endian mono PCM.

    Pure Python (array + math) so it works on 3.13+ where audioop is
    gone. 320 samples per call is cheap even on weak CPUs.
    """
    if not pcm:
        return 0.0
    samples = array.array("h")
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return 0.0
    samples.frombytes(pcm[:usable])
    n = len(samples)
    if not n:
        return 0.0
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / n)


class Vad:
    """Energy-based voice activity detector with hangover.

    - speech starts when `rms` exceeds `threshold` for `attack` frames
    - speech ends after `hangover` consecutive quiet frames
    - emits ("speech_start", t) / ("speech_end", t) events exactly once
      per utterance; ("frame", t, level) for instrumentation
    """

    def __init__(self, threshold: float = 350.0, hangover_ms: int = 400,
                 attack_ms: int = 40):
        self.threshold = threshold
        self.hangover = max(1, hangover_ms // FRAME_MS)
        self.attack = max(1, attack_ms // FRAME_MS)
        self._loud_run = 0
        self._quiet_run = 0
        self.in_speech = False

    def feed(self, pcm: bytes, t: float | None = None) -> list[tuple]:
        """Feed one FRAME_BYTES-sized frame; return any events."""
        t = t if t is not None else 0.0
        level = rms(pcm)
        events: list[tuple] = [("frame", t, round(level, 1))]

        if level >= self.threshold:
            self._loud_run += 1
            self._quiet_run = 0
        else:
            self._quiet_run += 1
            self._loud_run = 0

        if not self.in_speech and self._loud_run >= self.attack:
            self.in_speech = True
            events.append(("speech_start", t))
        elif self.in_speech and self._quiet_run >= self.hangover:
            self.in_speech = False
            events.append(("speech_end", t))
        return events
