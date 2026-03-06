import numpy as np


class InterfaceModel:
    PRESETS = {
        "ニュートラル・ライン出力": {
            "source_capacitance_pf": 80.0,
            "input_capacitance_pf": 180.0,
            "stage_bandwidth_hz": 280000.0,
            "settling_strength": 0.35,
            "stereo_ground_coupling": 0.25,
            "common_return_resistance": 0.03,
            "common_return_inductance": 0.35e-6,
            "stability_margin": 0.90,
            "surface_corner_hz": 6000.0,
            "shield_coverage": 0.92,
            "ingress_sensitivity": 0.18,
            "plug_contamination": 0.08,
            "contact_nonlinearity": 0.06,
        },
        "真空管プリ/ヴィンテージ": {
            "source_capacitance_pf": 140.0,
            "input_capacitance_pf": 240.0,
            "stage_bandwidth_hz": 160000.0,
            "settling_strength": 0.65,
            "stereo_ground_coupling": 0.45,
            "common_return_resistance": 0.08,
            "common_return_inductance": 0.90e-6,
            "stability_margin": 0.75,
            "surface_corner_hz": 5200.0,
            "shield_coverage": 0.78,
            "ingress_sensitivity": 0.42,
            "plug_contamination": 0.20,
            "contact_nonlinearity": 0.18,
        },
        "高速ソリッドステート": {
            "source_capacitance_pf": 45.0,
            "input_capacitance_pf": 120.0,
            "stage_bandwidth_hz": 550000.0,
            "settling_strength": 0.18,
            "stereo_ground_coupling": 0.16,
            "common_return_resistance": 0.02,
            "common_return_inductance": 0.20e-6,
            "stability_margin": 1.00,
            "surface_corner_hz": 7500.0,
            "shield_coverage": 0.96,
            "ingress_sensitivity": 0.10,
            "plug_contamination": 0.05,
            "contact_nonlinearity": 0.04,
        },
        "長尺RCAに敏感": {
            "source_capacitance_pf": 100.0,
            "input_capacitance_pf": 220.0,
            "stage_bandwidth_hz": 220000.0,
            "settling_strength": 0.75,
            "stereo_ground_coupling": 0.55,
            "common_return_resistance": 0.06,
            "common_return_inductance": 0.70e-6,
            "stability_margin": 0.72,
            "surface_corner_hz": 5800.0,
            "shield_coverage": 0.72,
            "ingress_sensitivity": 0.58,
            "plug_contamination": 0.30,
            "contact_nonlinearity": 0.24,
        },
        "安物/劣化RCA": {
            "source_capacitance_pf": 180.0,
            "input_capacitance_pf": 320.0,
            "stage_bandwidth_hz": 120000.0,
            "settling_strength": 0.92,
            "stereo_ground_coupling": 0.72,
            "common_return_resistance": 0.12,
            "common_return_inductance": 1.20e-6,
            "stability_margin": 0.62,
            "surface_corner_hz": 4800.0,
            "shield_coverage": 0.45,
            "ingress_sensitivity": 1.05,
            "plug_contamination": 0.62,
            "contact_nonlinearity": 0.58,
        },
    }

    def __init__(
        self,
        output_impedance=100.0,
        input_impedance=47000.0,
        source_capacitance_pf=80.0,
        input_capacitance_pf=180.0,
        stage_bandwidth_hz=280000.0,
        settling_strength=0.35,
        stereo_ground_coupling=0.25,
        common_return_resistance=0.03,
        common_return_inductance=0.35e-6,
        stability_margin=0.90,
        surface_corner_hz=6000.0,
        shield_coverage=0.92,
        ingress_sensitivity=0.18,
        plug_contamination=0.08,
        contact_nonlinearity=0.06,
    ):
        self.output_impedance = output_impedance
        self.input_impedance = input_impedance
        self.source_capacitance_pf = source_capacitance_pf
        self.input_capacitance_pf = input_capacitance_pf
        self.stage_bandwidth_hz = stage_bandwidth_hz
        self.settling_strength = settling_strength
        self.stereo_ground_coupling = stereo_ground_coupling
        self.common_return_resistance = common_return_resistance
        self.common_return_inductance = common_return_inductance
        self.stability_margin = stability_margin
        self.surface_corner_hz = surface_corner_hz
        self.shield_coverage = shield_coverage
        self.ingress_sensitivity = ingress_sensitivity
        self.plug_contamination = plug_contamination
        self.contact_nonlinearity = contact_nonlinearity

    @classmethod
    def from_preset(cls, preset_name, **overrides):
        params = dict(cls.PRESETS.get(preset_name, cls.PRESETS["ニュートラル・ライン出力"]))
        params.update(overrides)
        return cls(**params)

    def get_total_shunt_capacitance(self, cable_model):
        cable_cap = cable_model.get_total_capacitance()
        parasitic_cap = (self.source_capacitance_pf + self.input_capacitance_pf) * 1e-12
        return cable_cap + parasitic_cap

    def calculate_capacitive_stress(self, cable_model):
        total_cap_pf = self.get_total_shunt_capacitance(cable_model) * 1e12
        drive_factor = np.sqrt(max(self.output_impedance, 1e-6) / 100.0)
        return float(np.clip((total_cap_pf / 200.0) * drive_factor, 0.0, 12.0))

    def calculate_transfer_function(self, frequencies, cable_model):
        freqs = np.asarray(frequencies, dtype=np.float64)
        freqs = np.maximum(freqs, 1e-6)
        omega = 2.0 * np.pi * freqs
        s = 1j * omega

        material = cable_model.get_material_profile()
        total_cap = self.get_total_shunt_capacitance(cable_model)
        stress = self.calculate_capacitive_stress(cable_model)
        drive_resistance = max(self.output_impedance + 0.35 * cable_model.get_dc_series_resistance(), 1e-6)

        # 単純なRCだけでは見えにくい「長尺RCAの曇り」を、出力段の容量負荷ストレスと
        # セトリング成分として別途モデリングする。
        h_cap = 1.0 / (1.0 + s * drive_resistance * total_cap)

        effective_bw = self.stage_bandwidth_hz / (1.0 + 0.18 * stress * (1.15 - self.stability_margin))
        damping = 1.0 + 0.18 * stress * (1.20 - self.stability_margin)
        w0 = 2.0 * np.pi * max(effective_bw, 20000.0)
        h_stage = (w0 ** 2) / (s ** 2 + damping * w0 * s + w0 ** 2)

        settling_mix = np.clip(0.007 * stress * (0.70 + self.settling_strength), 0.0, 0.06)
        tau_fast = 7e-6 * (1.0 + 0.35 * stress)
        tau_slow = 55e-6 * (1.0 + 0.70 * stress)
        h_settle = (1.0 - settling_mix) + settling_mix * (
            0.70 / (1.0 + s * tau_fast) + 0.30 / (1.0 + s * tau_slow)
        )

        sheen_amount = 0.02 * material["surface_sheen"] * (0.85 + 0.04 * stress)
        w_shelf = 2.0 * np.pi * self.surface_corner_hz
        h_surface = 1.0 + sheen_amount * (s / (s + w_shelf))

        shield_loss = np.clip(
            0.08
            * self.ingress_sensitivity
            * (1.0 - self.shield_coverage)
            * (0.55 + 0.05 * stress)
            * (1.0 + 0.80 * self.plug_contamination),
            0.0,
            0.22,
        )
        shield_corner_hz = np.clip(2500.0 + 12000.0 * self.shield_coverage, 2500.0, 18000.0)
        h_shield = 1.0 - shield_loss * (s / (s + 2.0 * np.pi * shield_corner_hz))

        contact_loss = (
            0.010 * material["oxide_penalty"] * (0.55 + 0.10 * stress)
            + 0.035 * self.plug_contamination * (0.45 + 0.12 * stress)
        )
        w_contact = 2.0 * np.pi * 14000.0
        h_contact = 1.0 - contact_loss * (s / (s + w_contact))

        return h_cap * h_stage * h_settle * h_surface * h_shield * h_contact

    def estimate_shield_ingress_db(self, cable_model):
        stress = self.calculate_capacitive_stress(cable_model)
        material = cable_model.get_material_profile()
        shield_leak = np.clip(1.0 - self.shield_coverage, 0.0, 1.0)
        length_factor = 1.0 + 0.14 * cable_model.length
        source_factor = (max(self.output_impedance, 1.0) / 100.0) ** 0.35
        stress_factor = 0.55 + 0.06 * stress
        contamination_factor = 1.0 + 1.2 * self.plug_contamination + 0.4 * material["oxide_penalty"]
        ingress = np.clip(
            0.0012 * self.ingress_sensitivity * shield_leak * length_factor * source_factor * stress_factor * contamination_factor,
            1e-6,
            0.08,
        )
        return float(20.0 * np.log10(ingress))

    def estimate_contact_severity(self, cable_model):
        stress = self.calculate_capacitive_stress(cable_model)
        material = cable_model.get_material_profile()
        severity = (
            0.35 * self.plug_contamination
            + 0.25 * self.contact_nonlinearity
            + 0.15 * material["oxide_penalty"]
            + 0.015 * cable_model.length
        ) * (0.60 + 0.04 * stress)
        return float(np.clip(severity * 100.0, 0.0, 100.0))

    def get_shield_profile(self, cable_model):
        ingress_db = self.estimate_shield_ingress_db(cable_model)
        ingress_mix = float(np.clip(10.0 ** (ingress_db / 20.0), 0.0, 0.08))
        stress = self.calculate_capacitive_stress(cable_model)
        length = cable_model.length
        return {
            "haze_mix": np.clip(0.60 * ingress_mix * (1.0 + 0.04 * stress), 0.0, 0.03),
            "common_mode_mix": np.clip(0.90 * ingress_mix * (0.80 + 0.08 * length), 0.0, 0.04),
            "loss_mix": np.clip(2.80 * ingress_mix, 0.0, 0.08),
            "fast_tau": 12e-6 * (1.0 + 0.12 * length),
            "slow_tau": 220e-6 * (1.0 + 0.15 * length) * (1.0 + 0.05 * stress),
        }

    def get_contact_profile(self, cable_model):
        severity_norm = self.estimate_contact_severity(cable_model) / 100.0
        length = cable_model.length
        stress = self.calculate_capacitive_stress(cable_model)
        return {
            "dynamic_mix": np.clip(0.01 + 0.12 * severity_norm * (1.0 + 0.06 * length), 0.0, 0.18),
            "asymmetry": np.clip(0.15 + 0.80 * self.contact_nonlinearity + 0.40 * self.plug_contamination, 0.10, 1.50),
            "hf_tau": 8e-6 * (1.0 + 0.10 * length),
            "memory_tau": 70e-6 * (1.0 + 0.15 * length) * (1.0 + 0.05 * stress),
            "threshold": 0.015 + 0.04 * severity_norm,
        }

    def estimate_crosstalk_db(self, frequency_hz, cable_model):
        frequency_hz = max(float(frequency_hz), 1.0)
        stress = self.calculate_capacitive_stress(cable_model)
        material = cable_model.get_material_profile()
        z_return = np.abs(
            cable_model.get_return_impedance(frequency_hz)
            + self.common_return_resistance
            + 1j * 2.0 * np.pi * frequency_hz * self.common_return_inductance
        )
        scale = self.stereo_ground_coupling * (140.0 + 40.0 * stress) * (1.0 + 0.30 * material["oxide_penalty"])
        coupling = np.clip(scale * z_return / max(self.input_impedance, 1.0), 1e-7, 0.05)
        return float(20.0 * np.log10(coupling))

    def get_common_return_profile(self, cable_model):
        crosstalk_db = self.estimate_crosstalk_db(10000.0, cable_model)
        cross_mix = float(np.clip(10.0 ** (crosstalk_db / 20.0), 0.0, 0.02))
        length_factor = 1.0 + 0.18 * cable_model.length
        stress = self.calculate_capacitive_stress(cable_model)
        return {
            "ground_mix": np.clip(cross_mix * 0.75 * length_factor, 0.0, 0.03),
            "cross_mix": cross_mix,
            "inductive_mix": np.clip(cross_mix * 0.45 * (0.80 + cable_model.length / 6.0), 0.0, 0.02),
            "fast_tau": 18e-6 * (1.0 + 0.12 * cable_model.length),
            "slow_tau": 180e-6 * (1.0 + 0.22 * cable_model.length) * (1.0 + 0.08 * stress),
        }
