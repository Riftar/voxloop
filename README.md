# voxloop

A real-time voice turn engine — VAD, barge-in interruption, and
tools-while-talking — in pure stdlib Python. No audio device layer, no
models, no third-party packages, fully offline-testable.

The insight: interruption quality isn't a model problem, it's a state
machine problem. voxloop is that state machine, small enough to read in
one sitting and testable without a microphone or a GPU.

## Quick start

```bash
python3 scripts/make_fixtures.py   # generate the synthetic WAV fixtures
python3 tests/test_all.py          # 12 offline tests
```

## The engine

```
mic frames (20ms) -> Vad --speech_start--> [barge-in? cut playback NOW]
                          --speech_end---> turn: brain(text) [tools may
                                           run here] -> speak(reply)
```

- **Barge-in cuts first, asks later.** Interruption is decided on the
  frame where speech attack completes (40 ms default) and stops playback
  before any transcription runs. The sub-1s budget survives slow brains.
- **Tools never silence the mouth.** `brain()` runs in the turn path;
  `speak()` is non-blocking. A 3-second tool call delays the *reply*,
  never the *interruption handling*.
- **Pure-Python RMS** (`array` + `math`): no `audioop`, so it runs on
  Python 3.13+ unchanged.

## Fixtures (all synthetic, deterministic)

`speech_short.wav`, `speech_long.wav`, `silence.wav`, `noise.wav`
(sub-threshold hum), `convo_two_turns.wav` (speech/pause/speech/pause).

## Layout

- `vad.py` — RMS + energy VAD (attack/hangover, frame events)
- `engine.py` — `TurnEngine` state machine + synthetic PCM generators
- `scripts/make_fixtures.py` — builds the WAV fixtures
- `tests/test_all.py` — 12 offline tests
- `PLAN.md` — problem, goals, 3 checkpoints, rollback

## Honest ceiling

Energy VAD handles clean rooms; a noisy room wants a model (Silero).
That trade is documented in PLAN.md rather than hidden behind a flag.

## License

MIT
