import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import io
from cable_model import CableModel
from audio_processor import AudioProcessor

st.set_page_config(page_title="Audio System Chain Simulator", layout="wide")

if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = 44100

st.title("線で音が変わる？フルシステム・チェーン・シミュレーター")
st.markdown("""
**プレーヤー → [RCAケーブル] → アンプ → [スピーカーケーブル] → スピーカー**  
システム全体のケーブル影響を「2段階」でシミュレートします。
""")

# --- 設定セクション (2カラム) ---
col_line, col_spk = st.columns(2)

with col_line:
    st.header("1. 上流(Source/RCA)接続")
    st.info("プレーヤーからアンプ(またはDAP)への接続")
    
    l_material = st.selectbox("RCA 材質", list(CableModel.MATERIALS.keys()), key="l_mat")
    l_dielectric = st.selectbox("RCA 被膜", list(CableModel.DIELECTRICS.keys()), key="l_die")
    l_length = st.slider("RCA 長さ (m)", 0.1, 10.0, 1.0, key="l_len")
    l_diameter = st.slider("RCA 芯線径 (mm)", 0.01, 1.0, 0.3, step=0.01, key="l_dia")
    l_z_source = st.number_input("プレーヤー出力インピーダンス (Ω)", 0.1, 2000.0, 100.0, step=10.0)
    l_z_load = st.number_input("受け側入力インピーダンス (Ω)", 100.0, 100000.0, 10000.0, step=100.0)
    
    line_model = CableModel(
        length=l_length, 
        diameter=l_diameter * 1e-3, 
        spacing=1.5 * 1e-3,
        material=l_material,
        dielectric=l_dielectric,
        geometry="Coaxial",
        contact_res=0.01
    )

with col_spk:
    st.header("2. 下流(Cable/Driver)接続")
    preset_spk = st.selectbox("負荷プリセット", ["カスタム", "スピーカー (8Ω)", "ハイエンドIEM (Monarch MK3/20Ω)", "高インピーダンスHP (300Ω)"])
    
    if preset_spk == "スピーカー (8Ω)":
        s_z_source_def = 0.1; s_z_load_r_def = 8.0; s_z_load_l_def = 0.5; s_len_def = 3.0
    elif preset_spk == "ハイエンドIEM (Monarch MK3/20Ω)":
        s_z_source_def = 0.05; s_z_load_r_def = 20.0; s_z_load_l_def = 0.01; s_len_def = 1.2
    elif preset_spk == "高インピーダンスHP (300Ω)":
        s_z_source_def = 1.0; s_z_load_r_def = 300.0; s_z_load_l_def = 0.1; s_len_def = 3.0
    else:
        s_z_source_def = 0.1; s_z_load_r_def = 8.0; s_z_load_l_def = 0.5; s_len_def = 3.0

    s_material = st.selectbox("ケーブル 材質", list(CableModel.MATERIALS.keys()), key="s_mat")
    s_dielectric = st.selectbox("ケーブル 被膜", list(CableModel.DIELECTRICS.keys()), key="s_die")
    s_length = st.slider("ケーブル 長さ (m)", 0.1, 10.0, s_len_def, key="s_len")
    s_diameter = st.slider("ケーブル 線径 (mm)", 0.01, 5.0, 0.5, step=0.01, key="s_dia")
    s_z_source = st.number_input("アンプ/DAP出力インピーダンス (Ω)", 0.0, 100.0, s_z_source_def, step=0.01)
    s_z_load_r = st.number_input("ドライバー抵抗 (Ω)", 1.0, 1000.0, s_z_load_r_def)
    s_z_load_l_mh = st.number_input("ドライバー・インダクタンス (mH)", 0.0, 10.0, s_z_load_l_def)
    
    spk_model = CableModel(
        length=s_length, 
        diameter=s_diameter * 1e-3, 
        spacing=2.0 * 1e-3,
        material=s_material,
        dielectric=s_dielectric,
        geometry="Parallel",
        contact_res=0.005
    )

# --- 総合特性の計算 ---
st.divider()
st.subheader("システム解析 (System Analysis)")

fs_plot = np.logspace(1, 5, 500)
h_line = line_model.calculate_transfer_function(fs_plot, z_source=l_z_source, z_load_r=l_z_load, z_load_l=0)
h_spk = spk_model.calculate_transfer_function(fs_plot, z_source=s_z_source, z_load_r=s_z_load_r, z_load_l=0.5e-3)
h_total = h_line * h_spk

# ダンピングファクター (DF) の計算
# DF = Z_load / (Z_source + Z_cable)
r_cable_total = (spk_model.rho * spk_model.length / (np.pi * spk_model.radius**2)) * 2 + spk_model.contact_res
df = s_z_load_r / (s_z_source + r_cable_total)

col_res1, col_res2 = st.columns([2, 1])
with col_res2:
    st.metric("スピーカー側 ダンピングファクター", f"{df:.1f}", 
              help="値が高いほどアンプがスピーカーを正確に制動します。低いと低域がふくよか(ルーズ)になります。")
    st.write(f"ケーブル抵抗: {r_cable_total*1000:.1f} mΩ")

with col_res1:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # 振幅
    ax1.semilogx(fs_plot, 20 * np.log10(np.abs(h_total)), label="Total System", color="red", lw=2)
    ax1.set_title("Magnitude Response (Total)")
    ax1.set_ylabel("Gain (dB)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xlim(20, 20000)

    # 位相
    ax2.semilogx(fs_plot, np.angle(h_total, deg=True), color="orange")
    ax2.set_title("Phase Response (Total)")
    ax2.set_ylabel("Phase (deg)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_xlim(20, 20000)

    # インパルス応答 (過渡応答の可視化)
    proc_dummy = AudioProcessor(sample_rate=44100)
    ir_total = proc_dummy.generate_fir_filter(spk_model, n_taps=1024) # 簡易表示用
    t_ir = np.arange(len(ir_total)) / 44100 * 1000 # ms
    ax3.plot(t_ir, ir_total, color="cyan")
    ax3.set_title("Impulse Response (過渡応答 / 立ち上がり)")
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time (ms)")
    ax3.set_xlim(t_ir[len(ir_total)//2 - 20], t_ir[len(ir_total)//2 + 20]) # ピーク周辺を拡大
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

# --- 音声処理セクション ---
st.divider()
audio_source = st.radio("音声ソース", ["テスト用サインスイープ", "テスト用ホワイトノイズ", "アップロード"])

if audio_source == "アップロード":
    uploaded_file = st.file_uploader("WAVファイルをアップロードしてください", type=["wav"])
    if uploaded_file is not None:
        data, sr = sf.read(uploaded_file)
        st.session_state.audio_data = data
        st.session_state.sample_rate = sr
elif audio_source == "テスト用ホワイトノイズ":
    sr = 44100
    data = np.random.normal(0, 0.1, int(sr * 3.0))
    st.session_state.audio_data = data
    st.session_state.sample_rate = sr
else:
    sr = 44100
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    data = 0.5 * np.sin(2 * np.pi * 20 * ( (20000/20)**(t/duration) - 1 ) / ( np.log(20000/20)/duration ))
    st.session_state.audio_data = data
    st.session_state.sample_rate = sr

if st.session_state.audio_data is not None:
    if st.button("システム全体を通して再生"):
        with st.spinner("信号がチェーンを通過中..."):
            data = st.session_state.audio_data
            sr = st.session_state.sample_rate
            proc = AudioProcessor(sample_rate=sr)
            
            # --- Stage 1: RCA Line ---
            ir_line = proc.generate_fir_filter(line_model, z_source=l_z_source, z_load_r=l_z_load, z_load_l=0)
            from scipy.signal import fftconvolve
            if len(data.shape) > 1:
                processed = np.zeros_like(data)
                for i in range(data.shape[1]):
                    processed[:, i] = fftconvolve(data[:, i], ir_line, mode='same')
            else:
                processed = fftconvolve(data, ir_line, mode='same')
            processed = proc.apply_dielectric_absorption(processed, material_type=l_dielectric)
            
            # --- Stage 2: Speaker Cable ---
            ir_spk = proc.generate_fir_filter(spk_model, z_source=s_z_source, z_load_r=s_z_load_r, z_load_l=0.5e-3)
            if len(processed.shape) > 1:
                for i in range(processed.shape[1]):
                    processed[:, i] = fftconvolve(processed[:, i], ir_spk, mode='same')
            else:
                processed = fftconvolve(processed, ir_spk, mode='same')
            processed = proc.apply_dielectric_absorption(processed, material_type=s_dielectric)
            processed = proc.apply_thermal_modulation(processed, length=s_length, diameter=s_diameter)
            
            # 正規化
            max_val = np.max(np.abs(processed))
            if max_val > 0: processed /= max_val
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("オリジナル")
                buf_o = io.BytesIO(); sf.write(buf_o, data, sr, format='WAV')
                st.audio(buf_o.getvalue())
            with col2:
                st.write("システム通過後 (Total Chain)")
                buf_p = io.BytesIO(); sf.write(buf_p, processed, sr, format='WAV')
                st.audio(buf_p.getvalue())
