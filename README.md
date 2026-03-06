# Audio Cable & System Chain Simulator

[日本語版 README](./README.ja.md) | [Architecture](./ARCHITECTURE.md)

A research-oriented simulator for exploring how cable parameters, source/load
interaction, amplifier behavior, and time-domain artifacts can change an audio
system.

This project does not claim to have "solved" every cable-audibility question.
Its current contribution is more practical and modest:

- it provides an integrated model across cable physics, interface loading,
  amplifier behavior, DSP-side temporal effects, and perceptual-facing metrics
- it shows that clearly poor conditions, such as long or degraded cable runs,
  can measurably degrade system behavior
- it serves as a working platform for continued experiments

## Research Status

This repository should be read as:

- a simulator
- a hypothesis-generation tool
- an evolving research prototype

It should not yet be read as a complete proof of every listening impression.

## What It Models

- RLGC-style cable behavior with conductor, geometry, and dielectric parameters
- RCA line-stage interaction: source impedance, cable capacitance, settling,
  common return contamination, shield-quality stress, and contact penalties
- amplifier small-signal behavior: loop-margin stress, pole shift, phase margin,
  and frequency-dependent output impedance
- amplifier nonlinear behavior: slew-rate limiting, power sag, and harmonics
- speaker-side cable behavior with a richer load model for resonance and
  damping-related effects
- time-domain effects such as dielectric absorption, thermal modulation, raw
  impulse response, and step response

## Current Focus

The simulator is especially focused on two practical questions:

1. Why can a long or poor RCA cable sound more veiled than a simple RC roll-off
   would suggest?
2. Why can a very long speaker cable sound less forceful, less controlled, or
   less lively instead of merely quieter?

The current model can already express both trends under stressed conditions.

## Metrics Exposed in the App

- Magnitude / phase response
- Group delay
- Impulse response
- Step response
- TailRatio
- StageError
- RCA capacitance load
- Crosstalk estimate
- Shield ingress estimate
- Drive loss
- Minimum load impedance
- Damping factor
- Phase margin estimate
- `Zout @ 20 kHz`

## Project Structure

- `app.py`
  Thin Streamlit entrypoint kept at the repository root
- `src/audiofilter/app.py`
  Streamlit UI for controls, plots, and audio preview
- `src/audiofilter/system_chain.py`
  Shared analysis/playback signal-chain assembly
- `src/audiofilter/cable_model.py`
  RLGC cable model and material/dielectric helpers
- `src/audiofilter/interface_model.py`
  RCA interaction, settling, return contamination, shield/contact degradation
- `src/audiofilter/speaker_load_model.py`
  Synthetic complex speaker load model
- `src/audiofilter/amplifier_small_signal.py`
  Small-signal amplifier stability/output-impedance model
- `src/audiofilter/amplifier_model.py`
  Nonlinear amplifier model
- `src/audiofilter/audio_processor.py`
  FIR generation and time-domain emulation blocks
- `src/audiofilter/analysis_metrics.py`
  TailRatio, StageError, and step-response metrics

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Documentation

- [Japanese README](./README.ja.md)
- [Architecture](./ARCHITECTURE.md)

Internal working notes are kept under `private/` and are not intended as the
main public-facing documentation.

## Current Limitations

- The speaker load model is still synthetic, not measurement-based
- True stochastic RF ingress, hum, and intermittent contact faults are not yet
  fully modeled
- IEM/headphone paths are still simpler than the speaker path
- Real amplifier Bode measurements are not yet imported
- Listening impressions are not yet backed by a full controlled listening study

## Roadmap

- import measured impedance curves
- improve long/cheap RCA noise-ingress modeling
- add more physical loudspeaker motion/back-EMF behavior
- support measured amplifier response data
- continue validating model components against bench measurements

## License

This repository uses the Apache License 2.0. See [LICENSE](./LICENSE).
