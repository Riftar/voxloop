"""Synthesize the offline voice fixtures (stdlib wave module).

fixtures/
  speech_short.wav   1.0s tone burst  (a 'word')
  speech_long.wav    3.0s tone burst  (an 'utterance')
  silence.wav        2.0s zeros       (no speech)
  noise.wav          2.0s low-amp tone (below threshold = background hum)
  convo_two_turns.wav speech, pause, speech, pause (a scripted exchange)

All deterministic (fixed phase), so VAD tests are reproducible bit-for-bit.
"""

import array
import math
import os
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "fixtures")
RATE = 16000


def frames_for(seconds: float, gen) -> bytes:
    n = int(RATE * seconds)
    out = array.array("h", [int(gen(i)) for i in range(n)])
    return out.tobytes()


def voiced(i: int, amp: int = 6000, f: float = 220.0) -> float:
    return amp * math.sin(2 * math.pi * f * i / RATE)


def quiet(i: int, amp: int = 80, f: float = 110.0) -> float:
    return amp * math.sin(2 * math.pi * f * i / RATE)


def write(name: str, pcm: bytes) -> None:
    path = os.path.join(OUT, name)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    print(f"wrote {path} ({len(pcm)/RATE:.2f}s)")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    write("speech_short.wav", frames_for(1.0, voiced))
    write("speech_long.wav", frames_for(3.0, voiced))
    write("silence.wav", frames_for(2.0, lambda i: 0.0))
    write("noise.wav", frames_for(2.0, quiet))
    # scripted exchange: 1s speech, 0.6s pause, 1.5s speech, 0.6s pause
    convo = frames_for(1.0, voiced) + frames_for(0.6, lambda i: 0.0) \
        + frames_for(1.5, voiced) + frames_for(0.6, lambda i: 0.0)
    write("convo_two_turns.wav", convo)


if __name__ == "__main__":
    main()
