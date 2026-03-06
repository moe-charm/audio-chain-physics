import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import io
from scipy.signal import fftconvolve
from cable_model import CableModel
from audio_processor import AudioProcessor
from amplifier_model import AmplifierModel

DEFAULT_SAMPLE_RATE = 44100
FIR_TAPS = 2048
ANALYSIS_IR_SAMPLES = 8192


def sync_uploaded_audio():
    uploaded_file = st.session_state.get("uploaded_wav")
    if uploaded_file is None:
        if st.session_state.get("uploaded_audio_signature") is not None:
            st.session_state.audio_data = None
            st.session_state.sample_rate = DEFAULT_SAMPLE_RATE
            st.session_state.uploaded_audio_signature = None
        return

    signature = (uploaded_file.name, uploaded_file.size)
    if signature == st.session_state.get("uploaded_audio_signature"):
        return

    uploaded_file.seek(0)
    data, sr = sf.read(uploaded_file)
    st.session_state.audio_data = data
    st.session_state.sample_rate = sr
    st.session_state.uploaded_audio_signature = signature


def convolve_signal(data, impulse_response):
    if data.ndim > 1:
        processed = np.zeros_like(data, dtype=np.float64)
        for i in range(data.shape[1]):
            processed[:, i] = fftconvolve(data[:, i], impulse_response, mode="same")
        return processed
    return fftconvolve(data, impulse_response, mode="same")


def apply_amplifier(data, amplifier, sample_rate):
    if data.ndim > 1:
        processed = np.zeros_like(data, dtype=np.float64)
        for i in range(data.shape[1]):
            processed[:, i] = amplifier.process(data[:, i], sample_rate)
        return processed
    return amplifier.process(data, sample_rate)


def calculate_tail_ratio(ir_data, sample_rate):
    peak_idx = int(np.argmax(np.abs(ir_data)))
    peak_window = max(1, int(0.0001 * sample_rate))
    tail_window_end = max(peak_window + 1, int(0.005 * sample_rate))

    energy_peak = np.sum(ir_data[max(0, peak_idx - peak_window):peak_idx + peak_window] ** 2)
    energy_tail = np.sum(ir_data[peak_idx + peak_window:min(len(ir_data), peak_idx + tail_window_end)] ** 2)
    tail_ratio_db = 10 * np.log10(energy_tail / energy_peak) if (energy_peak > 0 and energy_tail > 0) else -100.0
    return peak_idx, tail_ratio_db


st.set_page_config(page_title="Audio System Chain Simulator v1.5", layout="wide")

if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = DEFAULT_SAMPLE_RATE
if 'audio_source' not in st.session_state:
    st.session_state.audio_source = "アップロード"
if 'uploaded_audio_signature' not in st.session_state:
    st.session_state.uploaded_audio_signature = None

sync_uploaded_audio()

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
        amp_type = st.selectbox("アンプ・プリセット", ["カスタム", "真空管(300B風)", "ハイスピード・トランジスタ", "現代的D級アンプ"])
        if amp_type == "真空管(300B風)":
            h2_def = 0.05; h3_def = 0.01; sr_def = 10.0; cap_def = 50; zout_def = 2.0
        elif amp_type == "ハイスピード・トランジスタ":
            h2_def = 0.0001; h3_def = 0.0001; sr_def = 100.0; cap_def = 500; zout_def = 0.01
        elif amp_type == "現代的D級アンプ":
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

load_inductance_h = z_l_l * 1e-3


def process_audio_chain(data, sample_rate, normalize_output=True):
    proc = AudioProcessor(sample_rate=sample_rate)
    processed = np.asarray(data, dtype=np.float64)

    ir_line = proc.generate_fir_filter(line_model, n_taps=FIR_TAPS, z_source=l_z_src, z_load_r=z_in_amp, z_load_l=0.0)
    processed = convolve_signal(processed, ir_line)
    processed = proc.apply_dielectric_absorption(processed, l_die)

    processed = apply_amplifier(processed, amp_model, sample_rate)

    ir_spk = proc.generate_fir_filter(
        spk_model,
        n_taps=FIR_TAPS,
        z_source=z_out_amp,
        z_load_r=z_l_r,
        z_load_l=load_inductance_h,
    )
    processed = convolve_signal(processed, ir_spk)
    processed = proc.apply_dielectric_absorption(processed, s_die)
    processed = proc.apply_thermal_modulation(processed, s_len, s_dia)

    if normalize_output:
        max_val = np.max(np.abs(processed))
        if max_val > 0:
            processed = processed / max_val

    return processed


analysis_sr = (
    st.session_state.sample_rate
    if st.session_state.audio_source == "アップロード" and st.session_state.audio_data is not None
    else DEFAULT_SAMPLE_RATE
)
plot_max_freq = min(20000.0, analysis_sr * 0.49)

with col_plot:
    st.header("📊 システム解析モニター")
    
    # 総合特性計算
    fs_plot = np.logspace(np.log10(10.0), np.log10(max(plot_max_freq, 20.0)), 500)
    h_line = line_model.calculate_transfer_function(fs_plot, z_source=l_z_src, z_load_r=z_in_amp, z_load_l=0)
    h_spk = spk_model.calculate_transfer_function(fs_plot, z_source=z_out_amp, z_load_r=z_l_r, z_load_l=load_inductance_h)
    h_total = h_line * h_spk

    # ダンピングファクターの計算と表示
    r_cable_total = (spk_model.rho * spk_model.length / (np.pi * spk_model.radius**2)) * 2 + spk_model.contact_res
    df = z_l_r / (z_out_amp + r_cable_total)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("スピーカー側 ダンピングファクター", f"{df:.1f}", help="制動力の指標。高いとタイト、低いとルーズな低域になります。")

    # グラフ描画
    fig, (ax1, ax2, ax_gd, ax3) = plt.subplots(4, 1, figsize=(12, 16))
    
    ax1.semilogx(fs_plot, 20 * np.log10(np.abs(h_total)), color="red", lw=2)
    ax1.set_title("周波数特性 (Magnitude Response)")
    ax1.set_ylabel("Gain (dB)"); ax1.grid(True, alpha=0.3); ax1.set_xlim(20, plot_max_freq)

    phase_rad = np.unwrap(np.angle(h_total))
    ax2.semilogx(fs_plot, np.rad2deg(phase_rad), color="orange")
    ax2.set_title("位相特性 (Phase Response)")
    ax2.set_ylabel("Phase (deg)"); ax2.grid(True, alpha=0.3); ax2.set_xlim(20, plot_max_freq)

    omega = 2 * np.pi * fs_plot
    gd = -np.gradient(phase_rad, omega) * 1e6
    ax_gd.semilogx(fs_plot, gd, color="purple")
    ax_gd.set_title("群遅延 (Group Delay) - 音場の正確さ指標")
    ax_gd.set_ylabel("Delay (μs)"); ax_gd.grid(True, alpha=0.3); ax_gd.set_xlim(20, plot_max_freq)
    gd_valid = gd[(fs_plot >= 20) & (fs_plot <= plot_max_freq)]
    if len(gd_valid) > 0: ax_gd.set_ylim(np.min(gd_valid) - 10, np.max(gd_valid) + 10)

    analysis_impulse = np.zeros(ANALYSIS_IR_SAMPLES)
    analysis_impulse[ANALYSIS_IR_SAMPLES // 2] = 1.0
    ir_total = process_audio_chain(analysis_impulse, analysis_sr, normalize_output=False)
    peak_idx, tail_ratio_db = calculate_tail_ratio(ir_total, analysis_sr)
    peak_val = np.max(np.abs(ir_total))
    ir_display = ir_total / peak_val if peak_val > 0 else ir_total
    t_ir = (np.arange(len(ir_display)) - peak_idx) / analysis_sr * 1000.0
    
    with col_m2:
        st.metric("音の質感 (TailRatio)", f"{tail_ratio_db:.1f} dB", help="波形の「尻尾」の量。値が高いほど『音が濃く/ふくよか』になります。")

    ax3.plot(t_ir, ir_display, color="cyan")
    ax3.set_title("過渡応答 (Impulse Response) - システム全体")
    ax3.set_xlim(-0.5, 2.0); ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Time (ms)")

    plt.tight_layout()
    st.pyplot(fig)

    # --- 再生セクション ---
    st.divider()
    st.subheader("🎵 試聴プレイヤー")
    audio_source = st.radio(
        "音声ソース",
        ["アップロード", "サインスイープ", "ホワイトノイズ"],
        horizontal=True,
        key="audio_source",
    )
    
    current_data = None
    current_sr = DEFAULT_SAMPLE_RATE
    if audio_source == "アップロード":
        st.file_uploader("WAVファイルをアップロード", type=["wav"], key="uploaded_wav")
        current_data = st.session_state.audio_data
        current_sr = st.session_state.sample_rate if current_data is not None else DEFAULT_SAMPLE_RATE
    elif audio_source == "ホワイトノイズ":
        sr = DEFAULT_SAMPLE_RATE; data = np.random.normal(0, 0.1, int(sr * 3.0))
        current_data = data; current_sr = sr
    else: # サインスイープ
        sr = DEFAULT_SAMPLE_RATE; t = np.linspace(0, 3.0, int(DEFAULT_SAMPLE_RATE * 3.0), endpoint=False)
        data = 0.5 * np.sin(2 * np.pi * 20 * ( (20000/20)**(t/3.0) - 1 ) / ( np.log(20000/20)/3.0 ))
        current_data = data; current_sr = sr

    if current_data is None and audio_source == "アップロード":
        st.warning("⚠️ WAVファイルをアップロードしてください。")
        st.button("▶ システム全体を通して再生 (聴き比べ)", use_container_width=True, disabled=True)
    else:
        if st.button("▶ システム全体を通して再生 (聴き比べ)", use_container_width=True):
            with st.spinner("信号がチェーンを通過中..."):
                data = current_data; sr = current_sr
                processed = process_audio_chain(data, sr, normalize_output=True)
                
                st.write("---")
                st.markdown("**▼ オリジナル (Original Source)**")
                buf_o = io.BytesIO(); sf.write(buf_o, data, sr, format='WAV'); st.audio(buf_o.getvalue(), format='audio/wav')
                st.markdown("**▼ システム通過後 (Processed Chain)**")
                buf_p = io.BytesIO(); sf.write(buf_p, processed, sr, format='WAV'); st.audio(buf_p.getvalue(), format='audio/wav')
                st.download_button("💾 加工済みファイルを保存", buf_p.getvalue(), "processed_audio.wav", "audio/wav", use_container_width=True)
