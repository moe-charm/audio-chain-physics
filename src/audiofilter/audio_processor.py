import numpy as np
from scipy.signal import lfilter

class AudioProcessor:
    DIELECTRIC_ABSORPTION_PROFILES = {
        "Teflon (PTFE)": [(0.00004, 35e-6), (0.00008, 180e-6), (0.00005, 0.9e-3)],
        "Polyethylene (PE)": [(0.00050, 55e-6), (0.00110, 280e-6), (0.00080, 1.8e-3)],
        "Polypropylene (PP)": [(0.00030, 45e-6), (0.00075, 220e-6), (0.00055, 1.5e-3)],
        "PVC (Vinyl)": [(0.00220, 80e-6), (0.00400, 550e-6), (0.00380, 3.2e-3)],
        "Rubber": [(0.00160, 70e-6), (0.00300, 420e-6), (0.00250, 2.6e-3)],
        "Cotton/Paper": [(0.00100, 60e-6), (0.00210, 360e-6), (0.00180, 2.2e-3)],
    }

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def generate_fir_from_frequency_response(self, h_f):
        h_f = np.asarray(h_f, dtype=np.complex128)
        h_f_full = np.concatenate([h_f, np.conj(h_f[1:-1][::-1])])
        ir = np.real(np.fft.ifft(h_f_full))
        # Playback uses a causal FIR so parameter changes are heard as changes in the
        # chain itself, not as fixed zero-phase pre-ringing from a centered impulse.
        taper_len = max(8, len(ir) // 16)
        taper = np.ones(len(ir), dtype=np.float64)
        taper[-taper_len:] = np.linspace(1.0, 0.0, taper_len)
        ir = ir * taper
        return ir

    def generate_fir_filter(self, model, n_taps=4096, z_source=0.1, z_load_r=8.0, z_load_l=0.5e-3):
        """
        CableModelからFIRフィルター(インパルス応答)を生成する (線形LTI部分)
        """
        freqs = np.linspace(0, self.sample_rate / 2, n_taps // 2 + 1)
        freqs[0] = 1e-6
        
        h_f = model.calculate_transfer_function(freqs, z_source=z_source, z_load_r=z_load_r, z_load_l=z_load_l)
        return self.generate_fir_from_frequency_response(h_f)

    def _resolve_cable_context(self, cable_or_material):
        if hasattr(cable_or_material, "dielectric_name"):
            return {
                "dielectric_name": cable_or_material.dielectric_name,
                "length": cable_or_material.length,
                "geometry": cable_or_material.geometry,
            }

        return {
            "dielectric_name": str(cable_or_material),
            "length": 1.0,
            "geometry": "Parallel",
        }

    def apply_dielectric_absorption(self, data, cable_or_material):
        """
        誘電吸収 (複数時定数のRC状態近似) のエミュレーション
        長尺RCAや損失の高い被膜ほど、エネルギーが複数の時間スケールで遅れて戻る。
        """
        context = self._resolve_cable_context(cable_or_material)
        dielectric_name = context["dielectric_name"]
        length = context["length"]
        geometry = context["geometry"]

        profile = self.DIELECTRIC_ABSORPTION_PROFILES.get(
            dielectric_name, self.DIELECTRIC_ABSORPTION_PROFILES["Polyethylene (PE)"]
        )

        length_scale = np.clip(0.70 + 0.35 * max(length - 1.0, 0.0), 0.70, 4.0)
        geometry_scale = 1.15 if geometry == "Coaxial" else 0.95
        mix_scale = length_scale * geometry_scale

        if len(data.shape) > 1:
            delayed_sum = np.zeros_like(data, dtype=np.float64)
            for base_mix, tau in profile:
                alpha = np.exp(-1.0 / (tau * self.sample_rate))
                for i in range(data.shape[1]):
                    delayed_sum[:, i] += (base_mix * mix_scale) * lfilter([1 - alpha], [1, -alpha], data[:, i])
        else:
            delayed_sum = np.zeros_like(data, dtype=np.float64)
            for base_mix, tau in profile:
                alpha = np.exp(-1.0 / (tau * self.sample_rate))
                delayed_sum += (base_mix * mix_scale) * lfilter([1 - alpha], [1, -alpha], data)

        total_mix = min(sum(base_mix for base_mix, _ in profile) * mix_scale, 0.18)
        return (np.asarray(data, dtype=np.float64) * (1.0 - total_mix)) + delayed_sum

    def apply_common_return_coupling(self, data, interface_model, cable_model):
        """
        ステレオ時の共通リターン/シールド汚染。
        L/Rのリターン電流が同じ基準電位を揺らすことで、像のにじみや軽い曇りを作る。
        """
        if len(data.shape) < 2 or data.shape[1] < 2:
            return data

        profile = interface_model.get_common_return_profile(cable_model)
        if profile["ground_mix"] <= 0 and profile["cross_mix"] <= 0 and profile["inductive_mix"] <= 0:
            return data

        stereo = np.array(data, dtype=np.float64, copy=True)
        left = stereo[:, 0]
        right = stereo[:, 1]

        alpha_fast = np.exp(-1.0 / (profile["fast_tau"] * self.sample_rate))
        alpha_slow = np.exp(-1.0 / (profile["slow_tau"] * self.sample_rate))

        def one_pole(signal, alpha):
            return lfilter([1 - alpha], [1, -alpha], signal)

        left_fast = one_pole(left, alpha_fast)
        right_fast = one_pole(right, alpha_fast)
        left_slow = one_pole(left, alpha_slow)
        right_slow = one_pole(right, alpha_slow)
        sum_slow = one_pole(left + right, alpha_slow)

        left_transient = left_fast - left_slow
        right_transient = right_fast - right_slow
        sum_transient = one_pole(left + right, alpha_fast) - sum_slow

        ground = profile["ground_mix"] * sum_slow + profile["inductive_mix"] * sum_transient

        stereo[:, 0] = left - ground - profile["cross_mix"] * (right_slow + 0.35 * right_transient)
        stereo[:, 1] = right - ground - profile["cross_mix"] * (left_slow + 0.35 * left_transient)
        return stereo

    def apply_shield_ingress(self, data, interface_model, cable_model):
        """
        シールド品質の低いRCAで起きやすい、低レベルのハゼ/被り。
        安価な長尺RCAの「抜けない」「曇る」を、軽い高域マスクとして近似する。
        """
        profile = interface_model.get_shield_profile(cable_model)
        if profile["haze_mix"] <= 0 and profile["common_mode_mix"] <= 0 and profile["loss_mix"] <= 0:
            return data

        alpha_fast = np.exp(-1.0 / (profile["fast_tau"] * self.sample_rate))
        alpha_slow = np.exp(-1.0 / (profile["slow_tau"] * self.sample_rate))

        def one_pole(signal, alpha):
            return lfilter([1 - alpha], [1, -alpha], signal)

        def make_haze(signal):
            fast = one_pole(signal, alpha_fast)
            slow = one_pole(signal, alpha_slow)
            transient = fast - slow
            haze = one_pole(np.tanh(8.0 * transient), alpha_fast)
            return haze

        if len(data.shape) > 1:
            stereo = np.array(data, dtype=np.float64, copy=True)
            common_haze = make_haze(np.sum(stereo, axis=1))
            for i in range(stereo.shape[1]):
                local_haze = make_haze(stereo[:, i])
                stereo[:, i] = (
                    stereo[:, i] * (1.0 - profile["loss_mix"])
                    + profile["haze_mix"] * local_haze
                    + profile["common_mode_mix"] * common_haze
                )
            return stereo

        signal = np.asarray(data, dtype=np.float64)
        local_haze = make_haze(signal)
        return signal * (1.0 - profile["loss_mix"]) + (profile["haze_mix"] + profile["common_mode_mix"]) * local_haze

    def apply_bad_contact_contamination(self, data, interface_model, cable_model):
        """
        劣化したプラグや接点の非線形損失。
        トランジェントが少し潰れ、薄い整流っぽい濁りが乗る。
        """
        profile = interface_model.get_contact_profile(cable_model)
        if profile["dynamic_mix"] <= 0:
            return data

        alpha_hf = np.exp(-1.0 / (profile["hf_tau"] * self.sample_rate))
        alpha_mem = np.exp(-1.0 / (profile["memory_tau"] * self.sample_rate))

        def one_pole(signal, alpha):
            return lfilter([1 - alpha], [1, -alpha], signal)

        def process_channel(signal):
            lowpassed = one_pole(signal, alpha_hf)
            transient = signal - lowpassed
            envelope = one_pole(np.abs(transient), alpha_mem)
            drive = envelope / (envelope + profile["threshold"])
            loss = profile["dynamic_mix"] * drive
            nonlinear = np.tanh(profile["asymmetry"] * (transient + 0.25 * np.sign(signal) * (signal ** 2)))
            return signal * (1.0 - loss) - 0.35 * loss * transient + 0.25 * loss * nonlinear

        if len(data.shape) > 1:
            out_data = np.zeros_like(data, dtype=np.float64)
            for i in range(data.shape[1]):
                out_data[:, i] = process_channel(data[:, i])
            return out_data

        return process_channel(np.asarray(data, dtype=np.float64))

    def apply_thermal_modulation(self, data, length, diameter):
        """
        熱による抵抗変調 (Thermal Modulation) のエミュレーション
        大電流が流れると温度が上がり抵抗が増える = 音量が微小に下がる (簡易コンプ)
        """
        # 線が細いほど、長いほど熱の影響を受けやすいと仮定
        base_resistance = length / (diameter**2) 
        thermal_coeff = np.clip(base_resistance * 1e-6, 0, 0.05) # 最大5%の変動
        
        # 音量のエンベロープ(RMS的)を計算 (時定数50ms)
        alpha = np.exp(-1.0 / (0.05 * self.sample_rate))
        
        def process_channel(chan_data):
            squared = chan_data**2
            # 簡易ローパスでエンベロープ（温度上昇）をシミュレート
            envelope = lfilter([1 - alpha], [1, -alpha], squared)
            # 温度が高いほどゲインが下がる
            gain_mod = 1.0 - (envelope * thermal_coeff)
            # ゲインが下がりすぎないようにクリップ
            gain_mod = np.clip(gain_mod, 0.8, 1.0)
            return chan_data * gain_mod

        if len(data.shape) > 1:
            out_data = np.zeros_like(data)
            for i in range(data.shape[1]):
                out_data[:, i] = process_channel(data[:, i])
            return out_data
        else:
            return process_channel(data)

