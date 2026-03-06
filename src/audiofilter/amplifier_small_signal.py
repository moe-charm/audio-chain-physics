import numpy as np


class AmplifierSmallSignalModel:
    PRESETS = {
        "カスタム": {
            "loop_gain_db": 40.0,
            "dominant_pole_hz": 18.0,
            "second_pole_hz": 65000.0,
            "load_interaction_hz": 190000.0,
            "compensation_zero_hz": 120000.0,
            "input_lowpass_hz": 180000.0,
            "open_loop_output_impedance": 6.0,
            "output_inductance_h": 0.7e-6,
            "capacitive_sensitivity": 0.70,
            "load_capacitance_ref_pf": 1800.0,
            "stability_margin": 0.90,
        },
        "真空管(300B風)": {
            "loop_gain_db": 28.0,
            "dominant_pole_hz": 22.0,
            "second_pole_hz": 26000.0,
            "load_interaction_hz": 82000.0,
            "compensation_zero_hz": 42000.0,
            "input_lowpass_hz": 110000.0,
            "open_loop_output_impedance": 16.0,
            "output_inductance_h": 1.4e-6,
            "capacitive_sensitivity": 1.15,
            "load_capacitance_ref_pf": 1300.0,
            "stability_margin": 0.72,
        },
        "ハイスピード・トランジスタ": {
            "loop_gain_db": 48.0,
            "dominant_pole_hz": 12.0,
            "second_pole_hz": 180000.0,
            "load_interaction_hz": 520000.0,
            "compensation_zero_hz": 320000.0,
            "input_lowpass_hz": 360000.0,
            "open_loop_output_impedance": 2.5,
            "output_inductance_h": 0.22e-6,
            "capacitive_sensitivity": 0.32,
            "load_capacitance_ref_pf": 2600.0,
            "stability_margin": 1.05,
        },
        "現代的D級アンプ": {
            "loop_gain_db": 38.0,
            "dominant_pole_hz": 18.0,
            "second_pole_hz": 54000.0,
            "load_interaction_hz": 150000.0,
            "compensation_zero_hz": 105000.0,
            "input_lowpass_hz": 190000.0,
            "open_loop_output_impedance": 5.0,
            "output_inductance_h": 0.85e-6,
            "capacitive_sensitivity": 0.66,
            "load_capacitance_ref_pf": 1800.0,
            "stability_margin": 0.88,
        },
    }

    def __init__(
        self,
        output_impedance=0.1,
        loop_gain_db=46.0,
        dominant_pole_hz=18.0,
        second_pole_hz=65000.0,
        load_interaction_hz=190000.0,
        compensation_zero_hz=120000.0,
        input_lowpass_hz=180000.0,
        open_loop_output_impedance=6.0,
        output_inductance_h=0.7e-6,
        capacitive_sensitivity=0.70,
        load_capacitance_ref_pf=1800.0,
        stability_margin=0.90,
    ):
        self.output_impedance = output_impedance
        self.loop_gain_db = loop_gain_db
        self.dominant_pole_hz = dominant_pole_hz
        self.second_pole_hz = second_pole_hz
        self.load_interaction_hz = load_interaction_hz
        self.compensation_zero_hz = compensation_zero_hz
        self.input_lowpass_hz = input_lowpass_hz
        self.open_loop_output_impedance = open_loop_output_impedance
        self.output_inductance_h = output_inductance_h
        self.capacitive_sensitivity = capacitive_sensitivity
        self.load_capacitance_ref_pf = load_capacitance_ref_pf
        self.stability_margin = stability_margin

    @classmethod
    def from_preset(cls, preset_name, **overrides):
        params = dict(cls.PRESETS.get(preset_name, cls.PRESETS["カスタム"]))
        params.update(overrides)
        return cls(**params)

    def _resolve_load_context(self, cable_model=None, z_load_r=8.0, z_load_l=0.5e-3):
        cable_cap_pf = cable_model.get_total_capacitance() * 1e12 if cable_model is not None else 0.0
        cable_inductance_h = cable_model.get_total_inductance() if cable_model is not None else 0.0

        load_factor = np.clip((8.0 / max(z_load_r, 0.25)) ** 0.35, 0.55, 2.50)
        reactive_factor = np.clip(1.0 + 900.0 * max(z_load_l, 0.0) + 0.18 * cable_inductance_h * 1e6, 1.0, 2.30)
        capacitive_stress = np.clip(
            (cable_cap_pf / max(self.load_capacitance_ref_pf, 1.0)) * load_factor * reactive_factor * self.capacitive_sensitivity,
            0.0,
            12.0,
        )

        stress_scale = max(0.20, 1.25 - self.stability_margin)
        dominant_pole_hz = self.dominant_pole_hz / (1.0 + 0.12 * capacitive_stress * stress_scale)
        second_pole_hz = self.second_pole_hz / (1.0 + 0.32 * capacitive_stress * stress_scale)
        load_pole_hz = self.load_interaction_hz / (1.0 + 0.68 * capacitive_stress * stress_scale)

        return {
            "cable_capacitance_pf": float(cable_cap_pf),
            "capacitive_stress": float(capacitive_stress),
            "dominant_pole_hz": float(dominant_pole_hz),
            "second_pole_hz": float(second_pole_hz),
            "load_pole_hz": float(load_pole_hz),
        }

    def _calculate_loop_shape(self, frequencies, context):
        freqs = np.asarray(frequencies, dtype=np.float64)
        freqs = np.maximum(freqs, 1e-6)
        s = 1j * 2.0 * np.pi * freqs
        loop_shape = 1.0 / (
            (1.0 + s / (2.0 * np.pi * context["dominant_pole_hz"]))
            * (1.0 + s / (2.0 * np.pi * context["second_pole_hz"]))
            * (1.0 + s / (2.0 * np.pi * context["load_pole_hz"]))
        )

        if self.compensation_zero_hz > 0:
            loop_shape *= 1.0 + s / (2.0 * np.pi * self.compensation_zero_hz)

        return loop_shape

    def _get_anchor_frequency(self, context):
        anchor_candidates = [
            5000.0,
            self.input_lowpass_hz * 0.12,
            context["second_pole_hz"] * 0.25,
            context["load_pole_hz"] * 0.33,
        ]
        anchor_hz = min(value for value in anchor_candidates if value > 0)
        return float(np.clip(anchor_hz, 1200.0, 5000.0))

    def calculate_loop_gain(self, frequencies, cable_model=None, z_load_r=8.0, z_load_l=0.5e-3):
        freqs = np.asarray(frequencies, dtype=np.float64)
        freqs = np.maximum(freqs, 1e-6)
        context = self._resolve_load_context(cable_model=cable_model, z_load_r=z_load_r, z_load_l=z_load_l)
        loop_shape = self._calculate_loop_shape(freqs, context)
        anchor_hz = self._get_anchor_frequency(context)
        anchor_shape = self._calculate_loop_shape(np.array([anchor_hz]), context)[0]
        target_loop_gain = 10.0 ** (self.loop_gain_db / 20.0)
        scale = target_loop_gain / max(np.abs(anchor_shape), 1e-12)
        loop_gain = scale * loop_shape
        context["loop_gain_anchor_hz"] = float(anchor_hz)
        context["loop_gain_anchor_db"] = float(self.loop_gain_db)
        return loop_gain, context

    def calculate_transfer_function(self, frequencies, cable_model=None, z_load_r=8.0, z_load_l=0.5e-3):
        freqs = np.asarray(frequencies, dtype=np.float64)
        freqs = np.maximum(freqs, 1e-6)
        s = 1j * 2.0 * np.pi * freqs
        loop_gain, _ = self.calculate_loop_gain(freqs, cable_model=cable_model, z_load_r=z_load_r, z_load_l=z_load_l)
        input_lowpass = 1.0 / (1.0 + s / (2.0 * np.pi * self.input_lowpass_hz))
        return input_lowpass * (loop_gain / (1.0 + loop_gain))

    def calculate_output_impedance(self, frequencies, cable_model=None, z_load_r=8.0, z_load_l=0.5e-3):
        freqs = np.asarray(frequencies, dtype=np.float64)
        freqs = np.maximum(freqs, 1e-6)
        omega = 2.0 * np.pi * freqs
        loop_gain, context = self.calculate_loop_gain(freqs, cable_model=cable_model, z_load_r=z_load_r, z_load_l=z_load_l)

        open_loop_delta = max(self.open_loop_output_impedance - self.output_impedance, 0.0)
        open_loop_delta *= 1.0 + 0.08 * context["capacitive_stress"]
        open_loop_output = open_loop_delta + 1j * omega * self.output_inductance_h
        return self.output_impedance + open_loop_output / (1.0 + loop_gain)

    def estimate_phase_margin(self, cable_model=None, z_load_r=8.0, z_load_l=0.5e-3):
        freqs = np.logspace(0, 7, 4000)
        loop_gain, _ = self.calculate_loop_gain(freqs, cable_model=cable_model, z_load_r=z_load_r, z_load_l=z_load_l)
        magnitudes = np.abs(loop_gain)
        crossings = np.where((magnitudes[:-1] >= 1.0) & (magnitudes[1:] < 1.0))[0]
        if len(crossings) > 0:
            idx = int(crossings[0] + 1)
        else:
            idx = int(np.argmin(np.abs(np.log10(magnitudes + 1e-12))))

        phase_margin_deg = 180.0 + np.angle(loop_gain[idx], deg=True)
        return float(np.clip(phase_margin_deg, -20.0, 120.0))

    def estimate_bandwidth(self, cable_model=None, z_load_r=8.0, z_load_l=0.5e-3):
        freqs = np.logspace(1, 7, 3000)
        response = self.calculate_transfer_function(freqs, cable_model=cable_model, z_load_r=z_load_r, z_load_l=z_load_l)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
        target = magnitude_db[0] - 3.0
        below = np.where(magnitude_db <= target)[0]
        if len(below) == 0:
            return float(freqs[-1])
        return float(freqs[int(below[0])])

    def get_diagnostics(self, cable_model=None, z_load_r=8.0, z_load_l=0.5e-3):
        context = self._resolve_load_context(cable_model=cable_model, z_load_r=z_load_r, z_load_l=z_load_l)
        output_impedance_20khz = self.calculate_output_impedance(
            np.array([20000.0]), cable_model=cable_model, z_load_r=z_load_r, z_load_l=z_load_l
        )[0]
        return {
            "cable_capacitance_pf": context["cable_capacitance_pf"],
            "capacitive_stress": context["capacitive_stress"],
            "phase_margin_deg": self.estimate_phase_margin(cable_model=cable_model, z_load_r=z_load_r, z_load_l=z_load_l),
            "output_impedance_20khz": float(np.abs(output_impedance_20khz)),
            "closed_loop_bandwidth_hz": self.estimate_bandwidth(cable_model=cable_model, z_load_r=z_load_r, z_load_l=z_load_l),
            "load_pole_shift_pct": float((1.0 - context["load_pole_hz"] / self.load_interaction_hz) * 100.0),
            "loop_gain_anchor_hz": self._get_anchor_frequency(context),
        }
