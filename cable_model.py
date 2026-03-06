import numpy as np
from scipy.constants import mu_0, epsilon_0

class CableModel:
    MATERIALS = {
        "Copper": 1.68e-8,
        "Silver": 1.59e-8,
        "Gold": 2.44e-8,
        "Aluminum": 2.82e-8
    }

    # 絶縁体の特性: (比誘電率 epsilon_r, 誘電正接 tan_delta)
    DIELECTRICS = {
        "Teflon (PTFE)": (2.1, 0.0001),  # 超低損失
        "Polyethylene (PE)": (2.3, 0.0005), # 一般的
        "Polypropylene (PP)": (2.2, 0.0003),
        "PVC (Vinyl)": (4.0, 0.05),     # 高損失・安価
        "Rubber": (3.0, 0.02),
        "Cotton/Paper": (1.5, 0.01)     # ヴィンテージ風
    }

    GEOMETRIES = ["Parallel", "Coaxial"]

    def __init__(self, length=1.0, diameter=1.0e-3, spacing=3.0e-3, 
                 material="Copper",
                 dielectric="Polyethylene (PE)",
                 geometry="Parallel",
                 contact_res=0.01):
        """
        ケーブルの物理パラメータを設定する
        :param contact_res: 接点抵抗 (Ω) (プラグの汚れや酸化を想定)
        """
        self.length = length
        self.diameter = diameter
        self.spacing = spacing
        self.material = material
        
        # 絶縁体設定
        dielectric_params = self.DIELECTRICS.get(dielectric, self.DIELECTRICS["Polyethylene (PE)"])
        self.epsilon_r = dielectric_params[0]
        self.tan_delta = dielectric_params[1]
        
        self.rho = self.MATERIALS.get(material, 1.68e-8)
        self.geometry = geometry
        self.radius = diameter / 2.0
        self.contact_res = contact_res
        
    def get_rlgc(self, frequencies):
        """
        周波数ごとの単位長あたりの RLGC パラメータを計算する
        """
        omega = 2 * np.pi * frequencies
        mu = mu_0
        
        # --- 1. 抵抗 R (表皮効果 + 近接効果) ---
        delta = np.sqrt(self.rho / (np.pi * frequencies * mu))
        r_dc = self.rho / (np.pi * self.radius**2)
        r_ac = (self.rho / (2 * np.pi * self.radius * delta))
        
        # 基本の交流抵抗 (Skin effect)
        R_skin = np.maximum(r_dc, r_ac)
        
        if self.geometry == "Parallel":
            # 近接効果 (Proximity Effect) の簡易モデル
            # 線間が近いほど、高域で抵抗が増大する
            # k_prox = 1 / sqrt(1 - (d/D)^2) 
            # (実際には周波数依存があるが、高域での増大をシミュレート)
            d_over_D = self.diameter / np.maximum(self.spacing, self.diameter + 1e-6)
            k_prox = 1.0 / np.sqrt(np.maximum(1.0 - d_over_D**2, 0.01))
            R = R_skin * 2 * k_prox # 往復2本分
        else:
            R = R_skin * 1.2 # 同軸の外部導体込みの簡易近似
            
        # --- 2 & 3. インダクタンス L と キャパシタンス C ---
        epsilon = epsilon_0 * self.epsilon_r
        effective_spacing = np.maximum(self.spacing, self.diameter + 1e-6)
        
        if self.geometry == "Parallel":
            l_ext = (mu / np.pi) * np.arccosh(effective_spacing / self.diameter)
            l_int = (mu / (8 * np.pi)) * 2 # 内部L
            L = np.full_like(frequencies, l_ext + l_int)
            c_val = (np.pi * epsilon) / np.arccosh(effective_spacing / self.diameter)
            C = np.full_like(frequencies, c_val)
        else:
            l_ext = (mu / (2 * np.pi)) * np.log(effective_spacing / self.diameter)
            l_int = mu / (8 * np.pi)
            L = np.full_like(frequencies, l_ext + l_int)
            c_val = (2 * np.pi * epsilon) / np.log(effective_spacing / self.diameter)
            C = np.full_like(frequencies, c_val)
        
        # --- 4. コンダクタンス G (誘電損失) ---
        # G = omega * C * tan_delta
        # これが「被膜による音のなまり」の主原因！
        G = omega * C * self.tan_delta
        
        return R, L, C, G

    def calculate_transfer_function(self, frequencies, z_source=0.1, z_load_r=8.0, z_load_l=0.5e-3):
        """
        伝達関数 H(f) を計算する (接点抵抗も考慮)
        """
        # 送信端に接点抵抗を足す
        z_source_total = z_source + self.contact_res
        
        omega = 2 * np.pi * frequencies
        R, L, C, G = self.get_rlgc(frequencies)
        
        z_load = z_load_r + 1j * omega * z_load_l
        
        gamma = np.sqrt((R + 1j * omega * L) * (G + 1j * omega * C))
        z0 = np.sqrt((R + 1j * omega * L) / (G + 1j * omega * C))
        
        gl = gamma * self.length
        A = np.cosh(gl)
        B = z0 * np.sinh(gl)
        C_abcd = (1/z0) * np.sinh(gl)
        D = A
        
        z_in = (A * z_load + B) / (C_abcd * z_load + D)
        v_source_div = z_in / (z_source_total + z_in)
        v_line_trans = 1 / (A + B / z_load)
        
        h_f = v_source_div * v_line_trans
        return h_f

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # テスト計算
    fs = np.logspace(1, 6, 500) # 10Hz to 1MHz
    model = CableModel(length=5.0, diameter=2.0e-3, spacing=10.0e-3) # 5m スピーカーケーブル
    h_f = model.calculate_transfer_function(fs, z_source=0.1, z_load=8.0)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.semilogx(fs, 20 * np.log10(np.abs(h_f)))
    plt.title("Cable Frequency Response (5m)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.semilogx(fs, np.angle(h_f, deg=True))
    plt.ylabel("Phase (deg)")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("cable_response_test.png")
    print("Test calculation finished. Plot saved as cable_response_test.png")
