import numpy as np
from scipy.signal import lfilter

class AmplifierModel:
    def __init__(self, 
                 input_impedance=47000, 
                 output_impedance=0.1,
                 slew_rate=50.0, # V/us
                 reference_peak_voltage=400.0,
                 capacitor_joules=100, # 電源コンデンサの「踏ん張り力」
                 harmonics_2nd=0.001, # 2次高調波 (真空管っぽさ)
                 harmonics_3rd=0.0005 # 3次高調波
                 ):
        self.zin = input_impedance
        self.zout = output_impedance
        self.sr = slew_rate
        self.reference_peak_voltage = reference_peak_voltage
        self.cap_j = capacitor_joules
        self.h2 = harmonics_2nd
        self.h3 = harmonics_3rd

    def apply_slew_rate_limit(self, data, fs):
        """
        スルーレート制限 (立ち上がりの鋭さ制限)
        """
        dt = 1.0 / fs
        max_delta_volts = self.sr * 1e6 * dt

        # 正規化波形を出力段の電圧スイングに写像してから制限する
        out_data = np.array(data, dtype=np.float64, copy=True)
        prev_voltage = out_data[0] * self.reference_peak_voltage
        for i in range(1, len(out_data)):
            target_voltage = out_data[i] * self.reference_peak_voltage
            delta = target_voltage - prev_voltage
            if abs(delta) > max_delta_volts:
                target_voltage = prev_voltage + np.sign(delta) * max_delta_volts
                out_data[i] = target_voltage / self.reference_peak_voltage
            prev_voltage = out_data[i] * self.reference_peak_voltage
        return out_data

    def apply_power_sag(self, data, fs, load_current=None, drive_stress=0.0):
        """
        電源サグ (大音量時に電圧が落ちて音が太く/歪む現象)
        """
        if load_current is None:
            demand = np.asarray(data, dtype=np.float64) ** 2
        else:
            demand = np.asarray(load_current, dtype=np.float64) ** 2

        alpha_fast = np.exp(-1.0 / (0.03 * fs))  # 30ms
        alpha_slow = np.exp(-1.0 / (0.18 * fs))  # 180ms
        energy_fast = lfilter([1 - alpha_fast], [1, -alpha_fast], demand)
        energy_slow = lfilter([1 - alpha_slow], [1, -alpha_slow], demand)
        energy = 0.65 * energy_fast + 0.35 * energy_slow

        stress_scale = 1.0 + 3.0 * np.clip(drive_stress, 0.0, 1.5)
        sag_factor = np.clip((energy * stress_scale) / (self.cap_j + 1e-6), 0, 0.35)
        gain_mod = 1.0 - sag_factor
        
        return data * gain_mod

    def apply_harmonics(self, data):
        """
        高調波歪みの追加
        """
        # y = x + h2*x^2 + h3*x^3
        # 正規化された入力に対して歪みを乗せる
        out = data + self.h2 * (data**2) + self.h3 * (data**3)
        return out

    def process(self, data, fs, load_current=None, drive_stress=0.0):
        """
        アンプの全処理を一括適用
        """
        # 1. スルーレート制限
        out = self.apply_slew_rate_limit(data, fs)
        # 2. 電源サグ
        out = self.apply_power_sag(out, fs, load_current=load_current, drive_stress=drive_stress)
        # 3. 高調波歪み
        out = self.apply_harmonics(out)
        
        # 最後にクリッピング
        return np.clip(out, -1.0, 1.0)
