import numpy as np


class SpeakerLoadModel:
    PRESETS = {
        "2-way Speaker (8Ω)": {
            "nominal_impedance": 8.0,
            "dc_resistance": 5.8,
            "voice_coil_inductance_h": 0.45e-3,
            "resonance_freq_hz": 48.0,
            "resonance_q": 1.05,
            "motional_peak_ohm": 28.0,
            "port_freq_hz": 0.0,
            "port_q": 1.20,
            "port_peak_ohm": 0.0,
            "crossover_freq_hz": 2200.0,
            "tweeter_impedance_ohm": 6.8,
            "tweeter_inductance_h": 0.06e-3,
            "tweeter_level": 1.0,
            "back_emf_strength": 0.42,
            "current_damping_sensitivity": 0.75,
        },
        "Long Cable Sensitive Speaker": {
            "nominal_impedance": 6.0,
            "dc_resistance": 4.4,
            "voice_coil_inductance_h": 0.62e-3,
            "resonance_freq_hz": 42.0,
            "resonance_q": 1.28,
            "motional_peak_ohm": 34.0,
            "port_freq_hz": 31.0,
            "port_q": 1.45,
            "port_peak_ohm": 18.0,
            "crossover_freq_hz": 1800.0,
            "tweeter_impedance_ohm": 5.4,
            "tweeter_inductance_h": 0.05e-3,
            "tweeter_level": 1.10,
            "back_emf_strength": 0.62,
            "current_damping_sensitivity": 1.05,
        },
        "Studio Monitor (4Ω)": {
            "nominal_impedance": 4.0,
            "dc_resistance": 3.2,
            "voice_coil_inductance_h": 0.35e-3,
            "resonance_freq_hz": 55.0,
            "resonance_q": 0.88,
            "motional_peak_ohm": 18.0,
            "port_freq_hz": 0.0,
            "port_q": 1.20,
            "port_peak_ohm": 0.0,
            "crossover_freq_hz": 2600.0,
            "tweeter_impedance_ohm": 4.4,
            "tweeter_inductance_h": 0.04e-3,
            "tweeter_level": 1.0,
            "back_emf_strength": 0.30,
            "current_damping_sensitivity": 0.88,
        },
    }

    def __init__(
        self,
        nominal_impedance=8.0,
        dc_resistance=5.8,
        voice_coil_inductance_h=0.45e-3,
        resonance_freq_hz=48.0,
        resonance_q=1.05,
        motional_peak_ohm=28.0,
        port_freq_hz=0.0,
        port_q=1.20,
        port_peak_ohm=0.0,
        crossover_freq_hz=2200.0,
        tweeter_impedance_ohm=6.8,
        tweeter_inductance_h=0.06e-3,
        tweeter_level=1.0,
        back_emf_strength=0.42,
        current_damping_sensitivity=0.75,
    ):
        self.nominal_impedance = nominal_impedance
        self.dc_resistance = dc_resistance
        self.voice_coil_inductance_h = voice_coil_inductance_h
        self.resonance_freq_hz = resonance_freq_hz
        self.resonance_q = resonance_q
        self.motional_peak_ohm = motional_peak_ohm
        self.port_freq_hz = port_freq_hz
        self.port_q = port_q
        self.port_peak_ohm = port_peak_ohm
        self.crossover_freq_hz = crossover_freq_hz
        self.tweeter_impedance_ohm = tweeter_impedance_ohm
        self.tweeter_inductance_h = tweeter_inductance_h
        self.tweeter_level = tweeter_level
        self.back_emf_strength = back_emf_strength
        self.current_damping_sensitivity = current_damping_sensitivity

    @classmethod
    def from_preset(cls, preset_name, **overrides):
        params = dict(cls.PRESETS.get(preset_name, cls.PRESETS["2-way Speaker (8Ω)"]))
        params.update(overrides)
        return cls(**params)

    def _resonance_peak(self, frequencies, f0_hz, q, peak_ohm):
        freqs = np.asarray(frequencies, dtype=np.float64)
        freqs = np.maximum(freqs, 1e-6)
        if f0_hz <= 0 or peak_ohm <= 0:
            return np.zeros_like(freqs, dtype=np.complex128)

        ratio = freqs / max(f0_hz, 1e-6)
        shape = 1.0 + 1j * q * (ratio - 1.0 / np.maximum(ratio, 1e-6))
        return peak_ohm / shape

    def calculate_impedance(self, frequencies):
        freqs = np.asarray(frequencies, dtype=np.float64)
        freqs = np.maximum(freqs, 1e-6)
        s = 1j * 2.0 * np.pi * freqs

        motional_peak = self.motional_peak_ohm * (0.65 + 0.85 * self.back_emf_strength)
        z_woofer = (
            self.dc_resistance
            + s * self.voice_coil_inductance_h
            + self._resonance_peak(freqs, self.resonance_freq_hz, self.resonance_q, motional_peak)
            + self._resonance_peak(freqs, self.port_freq_hz, self.port_q, self.port_peak_ohm)
        )

        if self.crossover_freq_hz > 0 and self.tweeter_level > 0:
            tweeter_resistance = self.tweeter_impedance_ohm / np.clip(self.tweeter_level, 0.2, 2.0)
            crossover_cap_f = 1.0 / (2.0 * np.pi * max(self.crossover_freq_hz, 20.0) * max(tweeter_resistance, 0.5))
            z_hp = 1.0 / np.maximum(s * crossover_cap_f, 1e-18)
            z_tweeter = tweeter_resistance + s * self.tweeter_inductance_h + z_hp
            z_total = 1.0 / (1.0 / z_woofer + 1.0 / z_tweeter)
        else:
            z_total = z_woofer

        real_part = np.maximum(np.real(z_total), 0.2)
        return real_part + 1j * np.imag(z_total)

    def get_reference_load(self):
        return float(self.nominal_impedance), float(self.voice_coil_inductance_h)

    def estimate_minimum_impedance(self):
        freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 500)
        return float(np.min(np.abs(self.calculate_impedance(freqs))))

    def estimate_drive_sensitivity(self, series_resistance_ohm):
        minimum_impedance = self.estimate_minimum_impedance()
        current_ratio = minimum_impedance / max(minimum_impedance + series_resistance_ohm, 1e-6)
        damping_loss = 1.0 - current_ratio
        return float(np.clip(damping_loss * self.current_damping_sensitivity, 0.0, 1.0))
