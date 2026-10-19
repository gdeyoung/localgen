# Music generation kit (fleet doctrine distillate)

Provenance: distilled 2026-10-19 from the fleet music kit
(`waypoint-music-kit-20261019`, sha256 `ce6bb125…d46f`, verbatim copy in the
private gdeyoung/hermes-fleet repo, `kits/`). The private kit additionally
carries operator taste profiles, 35 reference-song fingerprint analyses, and
internal routing details — deliberately not published here. Engine facts and
craft below are the reusable core.

## Engine map (what exists, what each is for)

| Lane | Engine | Role |
|---|---|---|
| Full vocal songs (primary) | Music3 (audio.cpp CLI, CUDA host) | production output |
| Score-driven songs | YuE2 (ABC scores + cot=melody) | compositional control; **CC BY-NC — non-commercial only** |
| Fast probes | ACE-Step 1.5 turbo (ComfyUI) | <10s identity probes, never keepers |
| Adaptive OST / stems | ACE-Step repaint/layer/complete | edits on existing audio |
| Ambient/drone | procedural WebAudio | after 2 failed AI attempts, go procedural |
| Transcription | Basic Pitch (audio→MIDI) | local |
| Reference analysis | chroma/tempogram/self-similarity | local |

The iGPU lane (this library's territory) maps to **ACE-Step turbo** for
probes/edits; full-song renders are CUDA-class. (See WORKFLOW.md §1 lanes.)

## What works (measured — keep)

- **ABC-first loop (YuE2):** compose an ABC score → lyric-fit pre-check →
  seed-race 2–3 renders → gate → fix the ABC, never re-prompt from scratch.
  The score is the iteration surface.
- **cot=melody doctrine:** rests-only vocal line in the score; melody mode
  sings the full lyric (30/30 pass). Full-lyric mode under-sings.
- **Quality gate per render:** source-separate (demucs 2-stem) → word-rate
  < 2.2 words/s + per-line coverage fuzzy-match + key check vs score + hiss
  check. Word-overlap % alone passes songs with dropped verses — retired.
- **Caption grammar (Music3):** anti-bleed structure; instrumental stem is
  the default keeper; bar-aligned seamless wrap; LUFS −17 for game use.
- **Master chain order:** de-hiss/tame-highs BEFORE loudnorm → genre-matched
  pitch correction on the separated vocal stem → LUFS −16 → stereo spread.
- **Lyric budgets:** 4 lines/verse, ~8 words/line, chant bridge, no empty
  section tags, no apostrophes, ~words/2 seconds at 120 wpm.
- **Blueprint-first:** reference DNA (2–3 songs) + harmonic map + motif plan
  + layer-entry table BEFORE any prompt/caption.
- **Speed:** 4–8 inference steps (defaults of 32 are overkill); check the
  `truncated` flag on every result.

## Dead ends (measured — do not retry)

- YuE2 "instrumental" via official recipes: model improvises vocals
  (150–200 invented words). Instrumentals → other engines, or separate the
  backing from a vocal render.
- Key changes by editing the ABC `K:` header only: doesn't transpose notes.
  Style-string key mentions do nothing. BPM header edits DO survive.
- Negative phrasing in style strings: negations get sung back.
- Renders > ~2.5 min drift; keepers exist only at max token budgets.
- Extended jazz chords / numbered degrees / tuplets: unsupported ABC cores.
- Seed lottery is real: same inputs flip sung/spoken across seeds — race
  seeds and gate each; don't chase a "better prompt."
- Engines have genre bell curves: switch engine after 2 failed batches.

## Fit with this library

localgen composes the video side (generate → interpolate → upscale) and the
voice side (cloned TTS); this kit is the soundtrack side. A fully local
one-call pipeline (video + score) is the open item — the fleet's engine
knowledge above is the starting map for porting an ACE-Step-class lane onto
the iGPU, at which point `finish` grows a score stage.
