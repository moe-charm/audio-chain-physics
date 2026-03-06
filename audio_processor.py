import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter

class AudioProcessor:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def generate_fir_filter(self, model, n_taps=4096, z_source=0.1, z_load_r=8.0, z_load_l=0.5e-3):
        """
        CableModelからFIRフィルター(インパルス応答)を生成する (線形LTI部分)
        """
        freqs = np.linspace(0, self.sample_rate / 2, n_taps // 2 + 1)
        freqs[0] = 1e-6
        
        h_f = model.calculate_transfer_function(freqs, z_source=z_source, z_load_r=z_load_r, z_load_l=z_load_l)
        h_f_full = np.concatenate([h_f, np.conj(h_f[1:-1][::-1])])
        ir = np.real(np.fft.ifft(h_f_full))
        ir = np.fft.fftshift(ir)
        ir = ir * np.hamming(len(ir))
        return ir

    def apply_dielectric_absorption(self, data, material_type):
        """
        誘電吸収 (RC状態近似) の簡易エミュレーション
        被膜に蓄えられた電荷が遅れて放出される現象 (微小な遅延ローパスとして実装)
        """
        # 材質による吸収係数の設定
        if "PVC" in material_type:
            da_mix = 0.05   # 吸収量が多い
            da_time_const = 0.01 # 放出が遅い (10ms)
        elif "PE" in material_type or "Polyethylene" in material_type:
            da_mix = 0.005  # 吸収量が少ない
            da_time_const = 0.002
        elif "Teflon" in material_type:
            da_mix = 0.0001 # ほぼゼロ
            da_time_const = 0.001
        else:
            da_mix = 0.01
            da_time_const = 0.005

        # 簡易的な 1極ローパス(IIR) で「ジワジワ戻る」成分を作る
        alpha = np.exp(-1.0 / (da_time_const * self.sample_rate))
        b = [1 - alpha]
        a = [1, -alpha]
        
        if len(data.shape) > 1:
            da_signal = np.zeros_like(data)
            for i in range(data.shape[1]):
                da_signal[:, i] = lfilter(b, a, data[:, i])
        else:
            da_signal = lfilter(b, a, data)
            
        # 原音から吸収分を引き、遅れて戻る成分を足す
        return (data * (1.0 - da_mix)) + (da_signal * da_mix)

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

