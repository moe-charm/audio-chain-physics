import numpy as np


def calculate_group_delay(freqs_hz, response):
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    response = np.asarray(response, dtype=np.complex128)
    phase_rad = np.unwrap(np.angle(response))
    omega = 2.0 * np.pi * freqs_hz
    group_delay_us = -np.gradient(phase_rad, omega) * 1e6
    return phase_rad, group_delay_us


def calculate_tail_ratio(ir_data, sample_rate):
    ir_data = np.asarray(ir_data, dtype=np.float64)
    peak_idx = int(np.argmax(np.abs(ir_data)))
    peak_window = max(1, int(0.0001 * sample_rate))
    tail_window_end = max(peak_window + 1, int(0.005 * sample_rate))

    energy_peak = np.sum(ir_data[max(0, peak_idx - peak_window):peak_idx + peak_window] ** 2)
    energy_tail = np.sum(ir_data[peak_idx + peak_window:min(len(ir_data), peak_idx + tail_window_end)] ** 2)
    tail_ratio_db = 10.0 * np.log10(energy_tail / energy_peak) if (energy_peak > 0 and energy_tail > 0) else -100.0
    return peak_idx, float(tail_ratio_db)


def calculate_stage_error(freqs_hz, response):
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    _, group_delay_us = calculate_group_delay(freqs_hz, response)
    audible = (freqs_hz >= 20.0) & (freqs_hz <= min(20000.0, np.max(freqs_hz)))
    if np.count_nonzero(audible) < 8:
        return 0.0

    x = np.log10(freqs_hz[audible])
    y = group_delay_us[audible]
    coeffs = np.polyfit(x, y, 1)
    trend = np.polyval(coeffs, x)
    excess_group_delay_us = y - trend
    return float(np.sqrt(np.mean(excess_group_delay_us ** 2)) / 1000.0)


def calculate_step_response(ir_data, peak_idx=None, pre_samples=16):
    ir_data = np.asarray(ir_data, dtype=np.float64)
    if peak_idx is None:
        peak_idx = int(np.argmax(np.abs(ir_data)))

    pre_samples = max(0, int(pre_samples))
    shift = max(peak_idx - pre_samples, 0)
    causal_ir = np.zeros_like(ir_data)
    causal_length = len(ir_data) - shift
    if causal_length > 0:
        causal_ir[:causal_length] = ir_data[shift:]
    return np.cumsum(causal_ir)


def calculate_step_metrics(step_response, sample_rate):
    step_response = np.asarray(step_response, dtype=np.float64)
    if len(step_response) == 0:
        return {"overshoot_pct": 0.0, "settling_ms": 0.0}

    tail_length = max(8, int(0.0005 * sample_rate))
    final_value = float(np.mean(step_response[-tail_length:]))
    if abs(final_value) < 1e-9:
        return {"overshoot_pct": 0.0, "settling_ms": 0.0}

    normalized = step_response / final_value
    overshoot_pct = max(0.0, (np.max(normalized) - 1.0) * 100.0)
    threshold = 0.02
    settling_idx = 0
    for idx in range(len(normalized) - 1, -1, -1):
        if abs(normalized[idx] - 1.0) > threshold:
            settling_idx = min(idx + 1, len(normalized) - 1)
            break

    settling_ms = settling_idx / sample_rate * 1000.0
    return {"overshoot_pct": float(overshoot_pct), "settling_ms": float(settling_ms)}


def calculate_difference_signal(reference, processed):
    reference = np.asarray(reference, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)

    if reference.shape != processed.shape:
        raise ValueError("reference and processed must have the same shape")

    if reference.ndim == 1:
        ref_energy = float(np.dot(reference, reference))
        gain = float(np.dot(processed, reference) / max(ref_energy, 1e-12))
        residual = processed - gain * reference
        return residual, gain

    residual = np.zeros_like(processed, dtype=np.float64)
    gains = []
    for channel_idx in range(reference.shape[1]):
        ref_channel = reference[:, channel_idx]
        proc_channel = processed[:, channel_idx]
        ref_energy = float(np.dot(ref_channel, ref_channel))
        gain = float(np.dot(proc_channel, ref_channel) / max(ref_energy, 1e-12))
        residual[:, channel_idx] = proc_channel - gain * ref_channel
        gains.append(gain)

    return residual, np.asarray(gains, dtype=np.float64)
