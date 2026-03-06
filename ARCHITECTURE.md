# Architecture

This document explains the public, high-level design of the simulator.

The main idea is to connect several layers that are often discussed
separately:

- cable physics
- interface loading
- amplifier behavior
- time-domain DSP approximation
- perceptual-facing analysis metrics

The result is not a claim of final truth. It is a structured research model
that can be inspected, modified, and tested.

## Design Goals

- keep analysis and playback on the same signal path
- make poor cable conditions measurable, not just anecdotal
- support both line-level and speaker-level scenarios
- expose interpretable metrics instead of only raw frequency response
- keep the system modular enough to swap in better models later

## Signal Chain

The current signal chain is:

1. source audio
2. line-stage cable transfer
3. line-stage dielectric absorption
4. common-return contamination
5. shield-ingress haze / bad-contact contamination
6. amplifier small-signal response
7. amplifier nonlinear processing
8. speaker cable transfer
9. speaker-side dielectric absorption
10. thermal modulation
11. final processed output

The same chain definition is used for:

- plotted analysis
- impulse/step response analysis
- audio preview

That shared assembly lives in `src/audiofilter/system_chain.py`.

## Module Responsibilities

### `app.py`

Thin repository-root entrypoint used for `streamlit run app.py`.

### `src/audiofilter/app.py`

Streamlit UI layer.

Responsibilities:

- user controls
- preset selection
- plot rendering
- audio preview
- wiring user parameters into the shared system chain

### `src/audiofilter/system_chain.py`

Central orchestration layer for the full modeled system.

Responsibilities:

- assemble the line stage, amplifier stage, and speaker stage
- build transfer responses for analysis
- build impulse responses for playback convolution
- ensure playback and analysis use the same underlying chain
- expose high-level metrics derived from the combined system

### `src/audiofilter/cable_model.py`

Physical cable model.

Responsibilities:

- RLGC-inspired transfer behavior
- material properties
- dielectric properties
- geometry-dependent parameters
- helper functions for total capacitance, series resistance, and return
  impedance

### `src/audiofilter/interface_model.py`

Line-level interaction model, mainly for RCA-style runs.

Responsibilities:

- source impedance x cable capacitance x input impedance interaction
- capacitive stress estimation
- settling/recovery shaping
- common-return coupling estimates
- shield-quality stress and ingress estimate
- contact degradation estimate

This is where the simulator tries to explain why a cheap or degraded long RCA
can sound more veiled than a simple treble shelf would suggest.

### `src/audiofilter/amplifier_small_signal.py`

Small-signal amplifier approximation.

Responsibilities:

- loop-margin stress
- pole-shift behavior under cable/load stress
- phase-margin estimate
- frequency-dependent output impedance estimate

This model is intentionally approximate, but it gives the cable/load model a
way to influence amplifier behavior rather than treating the amplifier as an
ideal voltage source.

### `src/audiofilter/amplifier_model.py`

Nonlinear amplifier approximation.

Responsibilities:

- slew-rate limiting
- power-supply sag
- current/drive-stress sensitivity
- harmonic generation

This stage is aimed at larger-signal behavior rather than linear response.

### `src/audiofilter/speaker_load_model.py`

Synthetic complex speaker-load model.

Responsibilities:

- low-frequency resonance
- optional port/secondary resonance
- inductive rise
- crossover-like branching
- sensitivity to series resistance and damping reduction

This is important because long speaker cables often feel like a loss of control
or grip, not only a level drop.

### `src/audiofilter/audio_processor.py`

Time-domain approximation layer.

Responsibilities:

- FIR generation from frequency response
- dielectric absorption emulation
- common-return contamination in stereo signals
- deterministic shield-ingress haze
- deterministic bad-contact contamination
- thermal modulation

Several of these are not literal first-principles simulations. They are compact
signal-processing approximations chosen to make the modeled mechanisms audible
and analyzable.

### `src/audiofilter/analysis_metrics.py`

Derived metric calculations.

Responsibilities:

- group delay
- StageError
- TailRatio
- step-response metrics

These metrics are used as bridges between raw transfer functions and listening
language such as density, smear, image stability, and drive.

## Analysis Philosophy

The simulator distinguishes between:

- physical causes
- observed behaviors
- perceptual-facing metrics

Examples:

- causes: capacitance loading, poor shielding, contact degradation, damping loss
- observed behaviors: HF attenuation, settling smear, crosstalk, overshoot
- metrics: TailRatio, StageError, drive loss, damping factor

This separation helps avoid treating every graph as an independent cause.

## Known Limitations

- the speaker load is still synthetic rather than measurement-driven
- stochastic RF ingress and hum are not yet modeled as true random processes
- bad-contact behavior is still a soft approximation, not an intermittent fault
  model
- IEM/headphone load models remain simpler than the speaker path
- measured amplifier Bode plots are not yet imported
- listening validation is still ongoing

## Why The Project Is Useful Already

Even with those limitations, the simulator is already useful because it can:

- demonstrate that clearly poor cable conditions can degrade behavior
- compare line-level and speaker-level mechanisms in one place
- show that some listening impressions are better explained by interaction
  effects than by cable resistance alone
- provide a reproducible sandbox for the next round of measurements

## Next Steps

- measured impedance import
- measured amplifier-response import
- better stochastic noise/ingress modeling
- richer speaker motion/back-EMF modeling
- stronger validation against bench measurements and controlled listening
