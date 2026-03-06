import numpy as np
from scipy.signal import lfilter

class AmplifierModel:
    def __init__(self, 
                 input_impedance=47000, 
                 output_impedance=0.1,
                 slew_rate=50.0, # V/us
                 capacitor_joules=100, # 電源コンデンサの「踏ん張り力」
                 harmonics_2nd=0.001, # 2次高調波 (真空管っぽさ)
                 harmonics_3rd=0.0005 # 3次高調波
                 ):
        self.zin = input_impedance
        self.zout = output_impedance
        self.sr = slew_rate
        self.cap_j = capacitor_joules
        self.h2 = harmonics_2nd
        self.h3 = harmonics_3rd

    def apply_slew_rate_limit(self, data, fs):
        """
        スルーレート制限 (立ち上がりの鋭さ制限)
        """
        dt = 1.0 / fs
        max_delta = self.sr * 1e6 * dt # 1サンプルあたりの最大電圧変化
        
        # 簡易的なクリップ処理でスルーレート制限を模擬
        out_data = np.copy(data)
        for i in range(1, len(out_data)):
            delta = out_data[i] - out_data[i-1]
            if abs(delta) > max_delta:
                out_data[i] = out_data[i-1] + np.sign(delta) * max_delta
        return out_data

    def apply_power_sag(self, data, fs):
        """
        電源サグ (大音量時に電圧が落ちて音が太く/歪む現象)
        """
        # 音圧のエンベロープ(エネルギー消費)を計算
        alpha = np.exp(-1.0 / (0.1 * fs)) # 100ms
        energy = lfilter([1 - alpha], [1, -alpha], data**2)
        
        # 電源の余裕度。コンデンサ量が多いほど、エネルギー消費による電圧降下が少ない
        sag_factor = np.clip(energy / (self.cap_j + 1e-6), 0, 0.2)
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

    def process(self, data, fs):
        """
        アンプの全処理を一括適用
        """
        # 1. スルーレート制限
        out = self.apply_slew_rate_limit(data, fs)
        # 2. 電源サグ
        out = self.apply_power_sag(out, fs)
        # 3. 高調波歪み
        out = self.apply_harmonics(out)
        
        # 最後にクリッピング
        return np.clip(out, -1.0, 1.0)
