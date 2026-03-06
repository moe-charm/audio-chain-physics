import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import io
from cable_model import CableModel
from audio_processor import AudioProcessor
from amplifier_model import AmplifierModel

st.set_page_config(page_title="Audio System Chain Simulator v1.5", layout="wide")

if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = 44100

st.title("オーディオ・システム・チェーン・シミュレーター v1.5")

# --- メインレイアウト (左に設定、右にグラフ) ---
col_config, col_plot = st.columns([1, 2])

with col_config:
    st.header("⚙️ システム構成")
    
    # [1] RCA
    with st.expander("🔌 [1] 上流ケーブル (RCA)", expanded=True):
        l_mat = st.selectbox("RCA 材質", list(CableModel.MATERIALS.keys()), key="l_mat")
        l_die = st.selectbox("RCA 被膜", list(CableModel.DIELECTRICS.keys()), key="l_die")
        l_len = st.slider("RCA 長さ (m)", 0.1, 10.0, 1.0, key="l_len")
        l_dia = st.slider("RCA 芯線径 (mm)", 0.01, 1.0, 0.3, step=0.01, key="l_dia")
        l_z_src = st.number_input("出力 Z (Ω)", 0.1, 2000.0, 100.0, key="l_zs")
        
        line_model = CableModel(length=l_len, diameter=l_dia*1e-3, spacing=1.5*1e-3,
                               material=l_mat, dielectric=l_die, geometry="Coaxial")

    # [2] Amplifier
    with st.expander("🎛️ [2] アンプ (Amplifier)", expanded=True):
        amp_type = st.selectbox("アンプ・プリセット", ["カスタム", "真空管(300B風)", "ハイスピード・トランジスタ", "プアオーディオD級"])
        if amp_type == "真空管(300B風)":
            h2_def = 0.05; h3_def = 0.01; sr_def = 10.0; cap_def = 50; zout_def = 2.0
        elif amp_type == "ハイスピード・トランジスタ":
            h2_def = 0.0001; h3_def = 0.0001; sr_def = 100.0; cap_def = 500; zout_def = 0.01
        elif amp_type == "プアオーディオD級":
            h2_def = 0.005; h3_def = 0.005; sr_def = 30.0; cap_def = 20; zout_def = 0.05
        else:
            h2_def = 0.001; h3_def = 0.001; sr_def = 50.0; cap_def = 100; zout_def = 0.1

        h2 = st.slider("2次高調波 (温かみ)", 0.0, 0.2, h2_def, step=0.001)
        h3 = st.slider("3次高調波 (輝き)", 0.0, 0.2, h3_def, step=0.001)
        sr_val = st.slider("スルーレート (V/us)", 1.0, 200.0, sr_def)
        cap_val = st.slider("電源余裕 (コンデンサ)", 1, 1000, cap_def)
        z_out_amp = st.number_input("出力インピーダンス (Ω)", 0.0, 10.0, zout_def)
        z_in_amp = st.number_input("入力インピーダンス (Ω)", 1000.0, 100000.0, 47000.0)

        amp_model = AmplifierModel(input_impedance=z_in_amp, output_impedance=z_out_amp,
                                 slew_rate=sr_val, capacitor_joules=cap_val,
                                 harmonics_2nd=h2, harmonics_3rd=h3)

    # [3] SPK/IEM Cable
    with st.expander("🧵 [3] 下流ケーブル (SPK/IEM)", expanded=True):
        s_mat = st.selectbox("SPK 材質", list(CableModel.MATERIALS.keys()), key="s_mat")
        s_die = st.selectbox("SPK 被膜", list(CableModel.DIELECTRICS.keys()), key="s_die")
        s_len = st.slider("SPK 長さ (m)", 0.1, 50.0, 3.0, key="s_len")
        s_dia = st.slider("SPK 線径 (mm)", 0.01, 5.0, 2.0, step=0.01, key="s_dia")
        
        spk_model = CableModel(length=s_len, diameter=s_dia*1e-3, spacing=5.0*1e-3,
                              material=s_mat, dielectric=s_die, geometry="Parallel")

    # [4] Final Load
    with st.expander("🔊 [4] 最終負荷 (Load)", expanded=False):
        load_preset = st.selectbox("負荷選択", ["Speaker (8Ω)", "IEM (20Ω)", "HP (300Ω)"])
        if load_preset == "Speaker (8Ω)": z_l_r = 8.0; z_l_l = 0.5
        elif load_preset == "IEM (20Ω)": z_l_r = 20.0; z_l_l = 0.01
        else: z_l_r = 300.0; z_l_l = 0.1
        z_l_r = st.number_input("抵抗 (Ω)", 1.0, 1000.0, z_l_r)
        z_l_l = st.number_input("インダクタンス (mH)", 0.0, 10.0, z_l_l)

with col_plot:
    st.header("📊 システム解析モニター")
    
    # 総合特性計算
    fs_plot = np.logspace(1, 5, 500)
    h_line = line_model.calculate_transfer_function(fs_plot, z_source=l_z_src, z_load_r=z_in_amp, z_load_l=0)
    h_spk = spk_model.calculate_transfer_function(fs_plot, z_source=z_out_amp, z_load_r=z_l_r, z_load_l=z_l_l*1e-3)
    h_total = h_line * h_spk

    # ダンピングファクターの計算と表示
    r_cable_total = (spk_model.rho * spk_model.length / (np.pi * spk_model.radius**2)) * 2 + spk_model.contact_res
    df = z_l_r / (z_out_amp + r_cable_total)
    st.metric("スピーカー側 ダンピングファクター", f"{df:.1f}", help="制動力の指標。高いとタイト、低いとルーズな低域になります。")

    # グラフ描画
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    ax1.semilogx(fs_plot, 20 * np.log10(np.abs(h_total)), color="red", lw=2)
    ax1.set_title("周波数特性 (Total Magnitude Response)")
    ax1.set_ylabel("Gain (dB)"); ax1.grid(True, alpha=0.3); ax1.set_xlim(20, 20000)

    ax2.semilogx(fs_plot, np.angle(h_total, deg=True), color="orange")
    ax2.set_title("位相特性 (Total Phase Response)")
    ax2.set_ylabel("Phase (deg)"); ax2.grid(True, alpha=0.3); ax2.set_xlim(20, 20000)

    proc_dummy = AudioProcessor(sample_rate=44100)
    ir_total = proc_dummy.generate_fir_filter(spk_model, n_taps=1024)
    t_ir = np.arange(len(ir_total)) / 44100 * 1000
    ax3.plot(t_ir, ir_total, color="cyan")
    ax3.set_title("過渡応答 (Impulse Response)")
    ax3.set_xlim(t_ir[512-20], t_ir[512+20]); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # --- 再生セクション (ここにソース選択も移動) ---
    st.divider()
    st.subheader("🎵 試聴プレイヤー")
    
    col_src, col_act = st.columns([1, 1])
    with col_src:
        audio_source = st.radio("音声ソース", ["サインスイープ", "ホワイトノイズ", "アップロード"], horizontal=True)
        if audio_source == "アップロード":
            uploaded_file = st.file_uploader("WAVファイルをアップロード", type=["wav"])
            if uploaded_file:
                data, sr = sf.read(uploaded_file)
                st.session_state.audio_data = data
                st.session_state.sample_rate = sr
        elif audio_source == "ホワイトノイズ":
            sr = 44100; data = np.random.normal(0, 0.1, int(sr * 3.0))
            st.session_state.audio_data = data; st.session_state.sample_rate = sr
        else: # サインスイープ
            sr = 44100; t = np.linspace(0, 3.0, int(44100 * 3.0))
            data = 0.5 * np.sin(2 * np.pi * 20 * ( (20000/20)**(t/3.0) - 1 ) / ( np.log(20000/20)/3.0 ))
            st.session_state.audio_data = data; st.session_state.sample_rate = sr

    with col_act:
        st.write(" ") # 余白
        if st.session_state.audio_data is not None:
            if st.button("▶ システム全体を通して再生 (聴き比べ)", use_container_width=True):
                with st.spinner("信号がチェーンを通過中..."):
                    data = st.session_state.audio_data; sr = st.session_state.sample_rate
                    proc = AudioProcessor(sample_rate=sr)
                    
                    # 1. RCA
                    ir_line = proc.generate_fir_filter(line_model, z_source=l_z_src, z_load_r=z_in_amp, z_load_l=0)
                    from scipy.signal import fftconvolve
                    if len(data.shape) > 1:
                        processed = np.zeros_like(data)
                        for i in range(data.shape[1]): processed[:, i] = fftconvolve(data[:, i], ir_line, mode='same')
                    else: processed = fftconvolve(data, ir_line, mode='same')
                    processed = proc.apply_dielectric_absorption(processed, l_die)
                    
                    # 2. Amp
                    if len(processed.shape) > 1:
                        for i in range(processed.shape[1]): processed[:, i] = amp_model.process(processed[:, i], sr)
                    else: processed = amp_model.process(processed, sr)
                    
                    # 3. SPK
                    ir_spk = proc.generate_fir_filter(spk_model, z_source=z_out_amp, z_load_r=z_l_r, z_load_l=z_l_l*1e-3)
                    if len(processed.shape) > 1:
                        for i in range(processed.shape[1]): processed[:, i] = fftconvolve(processed[:, i], ir_spk, mode='same')
                    else: processed = fftconvolve(processed, ir_spk, mode='same')
                    processed = proc.apply_dielectric_absorption(processed, s_die)
                    processed = proc.apply_thermal_modulation(processed, s_len, s_dia)
                    
                    # 正規化
                    max_val = np.max(np.abs(processed))
                    if max_val > 0: processed /= max_val
                    
                    col_o, col_p = st.columns(2)
                    with col_o: st.write("オリジナル"); buf_o = io.BytesIO(); sf.write(buf_o, data, sr, format='WAV'); st.audio(buf_o.getvalue())
                    with col_p: st.write("システム通過後"); buf_p = io.BytesIO(); sf.write(buf_p, processed, sr, format='WAV'); st.audio(buf_p.getvalue())
