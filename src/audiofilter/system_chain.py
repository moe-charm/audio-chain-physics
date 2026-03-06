from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .analysis_metrics import (
    calculate_group_delay,
    calculate_stage_error,
    calculate_step_metrics,
    calculate_step_response,
    calculate_tail_ratio,
)
from .audio_processor import AudioProcessor


def convolve_signal(data, impulse_response):
    if data.ndim > 1:
        processed = np.zeros_like(data, dtype=np.float64)
        for i in range(data.shape[1]):
            processed[:, i] = fftconvolve(data[:, i], impulse_response, mode="full")[: data.shape[0]]
        return processed
    return fftconvolve(data, impulse_response, mode="full")[: len(data)]


def apply_channelwise(data, processor):
    if data.ndim > 1:
        processed = np.zeros_like(data, dtype=np.float64)
        for i in range(data.shape[1]):
            processed[:, i] = processor(data[:, i])
        return processed
    return processor(data)


@dataclass
class LineStageConfig:
    cable_model: object
    interface_model: object
    source_impedance: float
    load_impedance: float


@dataclass
class PowerStageConfig:
    amplifier_nonlinear: object
    amplifier_small_signal: object
    cable_model: object
    load_resistance: float
    load_inductance_h: float
    load_model: object = None


@dataclass
class AudioSystemChain:
    line_stage: LineStageConfig
    power_stage: PowerStageConfig
    fir_taps: int = 2048
    analysis_ir_samples: int = 8192

    def _resolve_reference_load(self):
        if self.power_stage.load_model is not None:
            return self.power_stage.load_model.get_reference_load()
        return self.power_stage.load_resistance, self.power_stage.load_inductance_h

    def _resolve_load_impedance(self, frequencies):
        if self.power_stage.load_model is not None:
            return self.power_stage.load_model.calculate_impedance(frequencies)
        freqs = np.asarray(frequencies, dtype=np.float64)
        return self.power_stage.load_resistance + 1j * 2.0 * np.pi * freqs * self.power_stage.load_inductance_h

    def _estimate_drive_context(self):
        reference_load_r, reference_load_l = self._resolve_reference_load()
        cable_series_resistance = self.power_stage.cable_model.get_dc_series_resistance()
        amplifier_output_resistance = getattr(self.power_stage.amplifier_nonlinear, "zout", 0.0)
        total_series_resistance = cable_series_resistance + amplifier_output_resistance

        if self.power_stage.load_model is not None:
            minimum_impedance = self.power_stage.load_model.estimate_minimum_impedance()
            drive_loss = self.power_stage.load_model.estimate_drive_sensitivity(total_series_resistance)
        else:
            minimum_impedance = float(reference_load_r)
            current_ratio = minimum_impedance / max(minimum_impedance + total_series_resistance, 1e-6)
            drive_loss = float(np.clip(1.0 - current_ratio, 0.0, 1.0))

        effective_drive_impedance = max(0.60 * reference_load_r + 0.40 * minimum_impedance + total_series_resistance, 0.25)
        damping_factor = reference_load_r / max(total_series_resistance, 1e-9)

        return {
            "reference_load_ohm": float(reference_load_r),
            "reference_inductance_h": float(reference_load_l),
            "minimum_load_impedance_ohm": float(minimum_impedance),
            "cable_series_resistance_ohm": float(cable_series_resistance),
            "total_series_resistance_ohm": float(total_series_resistance),
            "effective_drive_impedance_ohm": float(effective_drive_impedance),
            "drive_loss": float(drive_loss),
            "damping_factor": float(damping_factor),
        }

    def _estimate_load_current_signal(self, signal):
        drive_context = self._estimate_drive_context()
        current_signal = np.asarray(signal, dtype=np.float64) / drive_context["effective_drive_impedance_ohm"]
        return current_signal, drive_context

    def build_line_stage_frequency_response(self, frequencies):
        h_cable = self.line_stage.cable_model.calculate_transfer_function(
            frequencies,
            z_source=self.line_stage.source_impedance,
            z_load_r=self.line_stage.load_impedance,
            z_load_l=0.0,
        )
        h_interface = self.line_stage.interface_model.calculate_transfer_function(
            frequencies,
            self.line_stage.cable_model,
        )
        return h_cable * h_interface

    def build_amplifier_frequency_response(self, frequencies):
        load_r, load_l = self._resolve_reference_load()
        return self.power_stage.amplifier_small_signal.calculate_transfer_function(
            frequencies,
            cable_model=self.power_stage.cable_model,
            z_load_r=load_r,
            z_load_l=load_l,
        )

    def build_speaker_cable_frequency_response(self, frequencies):
        load_r, load_l = self._resolve_reference_load()
        z_load = self._resolve_load_impedance(frequencies)
        z_source = self.power_stage.amplifier_small_signal.calculate_output_impedance(
            frequencies,
            cable_model=self.power_stage.cable_model,
            z_load_r=load_r,
            z_load_l=load_l,
        )
        return self.power_stage.cable_model.calculate_transfer_function(
            frequencies,
            z_source=z_source,
            z_load_r=load_r,
            z_load_l=load_l,
            z_load=z_load,
        )

    def build_total_frequency_response(self, frequencies):
        h_line = self.build_line_stage_frequency_response(frequencies)
        h_amp = self.build_amplifier_frequency_response(frequencies)
        h_spk = self.build_speaker_cable_frequency_response(frequencies)
        return {
            "line": h_line,
            "amp": h_amp,
            "speaker": h_spk,
            "total": h_line * h_amp * h_spk,
        }

    def _build_ir_from_response(self, proc, response):
        return proc.generate_fir_from_frequency_response(response)

    def build_line_stage_ir(self, proc):
        freqs = np.linspace(0, proc.sample_rate / 2, self.fir_taps // 2 + 1)
        freqs[0] = 1e-6
        return self._build_ir_from_response(proc, self.build_line_stage_frequency_response(freqs))

    def build_amplifier_ir(self, proc):
        freqs = np.linspace(0, proc.sample_rate / 2, self.fir_taps // 2 + 1)
        freqs[0] = 1e-6
        return self._build_ir_from_response(proc, self.build_amplifier_frequency_response(freqs))

    def build_speaker_cable_ir(self, proc):
        freqs = np.linspace(0, proc.sample_rate / 2, self.fir_taps // 2 + 1)
        freqs[0] = 1e-6
        return self._build_ir_from_response(proc, self.build_speaker_cable_frequency_response(freqs))

    def process_audio(self, data, sample_rate, normalize_output=True):
        proc = AudioProcessor(sample_rate=sample_rate)
        processed = np.asarray(data, dtype=np.float64)

        processed = convolve_signal(processed, self.build_line_stage_ir(proc))
        processed = proc.apply_dielectric_absorption(processed, self.line_stage.cable_model)
        processed = proc.apply_common_return_coupling(
            processed,
            self.line_stage.interface_model,
            self.line_stage.cable_model,
        )
        processed = proc.apply_shield_ingress(
            processed,
            self.line_stage.interface_model,
            self.line_stage.cable_model,
        )
        processed = proc.apply_bad_contact_contamination(
            processed,
            self.line_stage.interface_model,
            self.line_stage.cable_model,
        )

        processed = convolve_signal(processed, self.build_amplifier_ir(proc))
        current_signal, drive_context = self._estimate_load_current_signal(processed)
        if processed.ndim > 1:
            nonlinear = np.zeros_like(processed, dtype=np.float64)
            for i in range(processed.shape[1]):
                nonlinear[:, i] = self.power_stage.amplifier_nonlinear.process(
                    processed[:, i],
                    sample_rate,
                    load_current=current_signal[:, i],
                    drive_stress=drive_context["drive_loss"],
                )
            processed = nonlinear
        else:
            processed = self.power_stage.amplifier_nonlinear.process(
                processed,
                sample_rate,
                load_current=current_signal,
                drive_stress=drive_context["drive_loss"],
            )

        processed = convolve_signal(processed, self.build_speaker_cable_ir(proc))
        processed = proc.apply_dielectric_absorption(processed, self.power_stage.cable_model)
        processed = proc.apply_thermal_modulation(
            processed,
            self.power_stage.cable_model.length,
            self.power_stage.cable_model.diameter * 1e3,
        )

        if normalize_output:
            max_val = np.max(np.abs(processed))
            if max_val > 0:
                processed = processed / max_val

        return processed

    def analyze(self, sample_rate, plot_max_freq=20000.0, plot_points=500):
        plot_max_freq = min(20000.0, plot_max_freq)
        freqs = np.logspace(np.log10(10.0), np.log10(max(plot_max_freq, 20.0)), plot_points)
        responses = self.build_total_frequency_response(freqs)
        phase_rad, group_delay_us = calculate_group_delay(freqs, responses["total"])
        stage_error_ms = calculate_stage_error(freqs, responses["total"])
        drive_context = self._estimate_drive_context()

        analysis_impulse = np.zeros(self.analysis_ir_samples)
        analysis_impulse[self.analysis_ir_samples // 2] = 1.0
        ir_total = self.process_audio(analysis_impulse, sample_rate, normalize_output=False)
        peak_idx, tail_ratio_db = calculate_tail_ratio(ir_total, sample_rate)
        peak_val = np.max(np.abs(ir_total))
        ir_display = ir_total / peak_val if peak_val > 0 else ir_total
        t_ir_ms = (np.arange(len(ir_display)) - peak_idx) / sample_rate * 1000.0

        step_response = calculate_step_response(ir_display, peak_idx=peak_idx)
        step_metrics = calculate_step_metrics(step_response, sample_rate)
        step_tail = max(8, int(0.0005 * sample_rate))
        step_final = np.mean(step_response[-step_tail:]) if len(step_response) > 0 else 1.0
        if abs(step_final) > 1e-9:
            step_display = step_response / step_final
        else:
            step_display = step_response
        t_step_ms = np.arange(len(step_display)) / sample_rate * 1000.0 - (16 / sample_rate * 1000.0)

        amp_diagnostics = self.power_stage.amplifier_small_signal.get_diagnostics(
            cable_model=self.power_stage.cable_model,
            z_load_r=self._resolve_reference_load()[0],
            z_load_l=self._resolve_reference_load()[1],
        )

        return {
            "freqs_hz": freqs,
            "responses": responses,
            "phase_rad": phase_rad,
            "group_delay_us": group_delay_us,
            "stage_error_ms": stage_error_ms,
            "ir_total": ir_total,
            "ir_display": ir_display,
            "t_ir_ms": t_ir_ms,
            "tail_ratio_db": tail_ratio_db,
            "step_display": step_display,
            "t_step_ms": t_step_ms,
            "step_metrics": step_metrics,
            "line_capacitance_pf": self.line_stage.interface_model.get_total_shunt_capacitance(self.line_stage.cable_model) * 1e12,
            "line_crosstalk_db": self.line_stage.interface_model.estimate_crosstalk_db(10000.0, self.line_stage.cable_model),
            "line_ingress_db": self.line_stage.interface_model.estimate_shield_ingress_db(self.line_stage.cable_model),
            "line_contact_severity_pct": self.line_stage.interface_model.estimate_contact_severity(self.line_stage.cable_model),
            "line_stress": self.line_stage.interface_model.calculate_capacitive_stress(self.line_stage.cable_model),
            "amp_diagnostics": amp_diagnostics,
            "drive_loss_pct": drive_context["drive_loss"] * 100.0,
            "load_min_impedance_ohm": drive_context["minimum_load_impedance_ohm"],
            "power_series_resistance_ohm": drive_context["total_series_resistance_ohm"],
            "damping_factor": drive_context["damping_factor"],
        }
