import tempfile
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

from .amplifier_model import AmplifierModel
from .amplifier_small_signal import AmplifierSmallSignalModel
from .cable_model import CableModel
from .interface_model import InterfaceModel
from .speaker_load_model import SpeakerLoadModel
from .system_chain import AudioSystemChain, LineStageConfig, PowerStageConfig

DEFAULT_SAMPLE_RATE = 44100
FIR_TAPS = 2048
ANALYSIS_IR_SAMPLES = 8192
PREVIEW_DIR = Path(tempfile.gettempdir()) / "audio_chain_physics_previews"

NONLINEAR_AMP_PRESETS = {
    "カスタム": {"h2": 0.001, "h3": 0.001, "slew_rate": 50.0, "capacitor_joules": 100, "output_impedance": 0.1},
    "真空管(300B風)": {"h2": 0.05, "h3": 0.01, "slew_rate": 10.0, "capacitor_joules": 50, "output_impedance": 2.0},
    "ハイスピード・トランジスタ": {"h2": 0.0001, "h3": 0.0001, "slew_rate": 100.0, "capacitor_joules": 500, "output_impedance": 0.01},
    "現代的D級アンプ": {"h2": 0.005, "h3": 0.005, "slew_rate": 30.0, "capacitor_joules": 20, "output_impedance": 0.05},
}

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


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


def get_audio_source_data(audio_source):
    if audio_source == "アップロード":
        st.file_uploader("WAVファイルをアップロード", type=["wav"], key="uploaded_wav")
        current_data = st.session_state.audio_data
        current_sr = st.session_state.sample_rate if current_data is not None else DEFAULT_SAMPLE_RATE
        return current_data, current_sr

    if audio_source == "ホワイトノイズ":
        sr = DEFAULT_SAMPLE_RATE
        return np.random.normal(0, 0.1, int(sr * 3.0)), sr

    sr = DEFAULT_SAMPLE_RATE
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    sweep = 0.5 * np.sin(2 * np.pi * 20 * (((20000 / 20) ** (t / 3.0)) - 1) / (np.log(20000 / 20) / 3.0))
    return sweep, sr


def audio_to_wav_file(data, sample_rate, label):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    safe = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    path = PREVIEW_DIR / f"{label}_{uuid.uuid4().hex}.wav"
    sf.write(path, safe, sample_rate, format="WAV", subtype="PCM_16")
    return str(path)


def cleanup_preview_files(payload):
    if not payload:
        return
    for key in ("original_path", "processed_path", "difference_path"):
        path = payload.get(key)
        if not path:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def build_preview_signature(audio_source, current_sr, current_data, **settings):
    data_shape = tuple(current_data.shape) if current_data is not None else None
    return (
        audio_source,
        current_sr,
        data_shape,
        tuple(sorted(settings.items())),
        st.session_state.get("uploaded_audio_signature"),
    )


st.set_page_config(page_title="Audio System Chain Simulator v1.7", layout="wide")

if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "sample_rate" not in st.session_state:
    st.session_state.sample_rate = DEFAULT_SAMPLE_RATE
if "audio_source" not in st.session_state:
    st.session_state.audio_source = "アップロード"
if "uploaded_audio_signature" not in st.session_state:
    st.session_state.uploaded_audio_signature = None
if "preview_payload" not in st.session_state:
    st.session_state.preview_payload = None

sync_uploaded_audio()

st.title("オーディオ・システム・チェーン・シミュレーター v1.7")

col_config, col_plot = st.columns([1, 2])

with col_config:
    st.header("⚙️ システム構成")

    with st.expander("🔌 [1] 上流ケーブル (RCA)", expanded=True):
        l_mat = st.selectbox("RCA 材質", list(CableModel.MATERIALS.keys()), key="l_mat")
        l_die = st.selectbox("RCA 被膜", list(CableModel.DIELECTRICS.keys()), key="l_die")
        l_len = st.slider("RCA 長さ (m)", 0.1, 10.0, 1.0, key="l_len")
        l_dia = st.slider("RCA 芯線径 (mm)", 0.01, 1.0, 0.3, step=0.01, key="l_dia")
        l_z_src = st.number_input("出力 Z (Ω)", 0.1, 2000.0, 100.0, key="l_zs")
        l_contact_res = st.number_input("RCA 接点抵抗 (Ω)", 0.0, 0.2, 0.01, step=0.001, format="%.3f")

        line_model = CableModel(
            length=l_len,
            diameter=l_dia * 1e-3,
            spacing=1.5e-3,
            material=l_mat,
            dielectric=l_die,
            geometry="Coaxial",
            contact_res=l_contact_res,
        )

        with st.expander("🧪 RCAインターフェース詳細", expanded=False):
            line_stage_preset = st.selectbox("出力段プリセット", list(InterfaceModel.PRESETS.keys()), key="line_stage_preset")
            line_stage_defaults = InterfaceModel.PRESETS[line_stage_preset]
            st.caption("長尺RCAや機器差をここで追い込めます。")
            source_cap_pf = st.slider("出力側浮遊容量 (pF)", 10.0, 400.0, float(line_stage_defaults["source_capacitance_pf"]), step=5.0)
            input_cap_pf = st.slider("入力側容量 (pF)", 20.0, 600.0, float(line_stage_defaults["input_capacitance_pf"]), step=10.0)
            stage_bw_khz = st.slider("出力段帯域 (kHz)", 50, 1000, int(line_stage_defaults["stage_bandwidth_hz"] / 1000.0), step=10)
            settling_strength = st.slider("容量負荷セトリング感度", 0.0, 1.0, float(line_stage_defaults["settling_strength"]), step=0.01)
            stereo_ground_coupling = st.slider("共通GND結合", 0.0, 1.0, float(line_stage_defaults["stereo_ground_coupling"]), step=0.01)
            shield_coverage = st.slider("シールド品質", 0.0, 1.0, float(line_stage_defaults["shield_coverage"]), step=0.01)
            ingress_sensitivity = st.slider("RF侵入感度", 0.0, 1.5, float(line_stage_defaults["ingress_sensitivity"]), step=0.01)
            plug_contamination = st.slider("プラグ汚れ/接点劣化", 0.0, 1.0, float(line_stage_defaults["plug_contamination"]), step=0.01)
            contact_nonlinearity = st.slider("接点非線形", 0.0, 1.0, float(line_stage_defaults["contact_nonlinearity"]), step=0.01)

    with st.expander("🎛️ [2] アンプ (Amplifier)", expanded=True):
        amp_type = st.selectbox("アンプ・プリセット", list(NONLINEAR_AMP_PRESETS.keys()))
        nonlinear_defaults = NONLINEAR_AMP_PRESETS[amp_type]
        small_signal_defaults = AmplifierSmallSignalModel.PRESETS.get(amp_type, AmplifierSmallSignalModel.PRESETS["カスタム"])

        h2 = st.slider("2次高調波 (温かみ)", 0.0, 0.2, nonlinear_defaults["h2"], step=0.001)
        h3 = st.slider("3次高調波 (輝き)", 0.0, 0.2, nonlinear_defaults["h3"], step=0.001)
        sr_val = st.slider("スルーレート (V/us)", 1.0, 200.0, nonlinear_defaults["slew_rate"])
        cap_val = st.slider("電源余裕 (コンデンサ)", 1, 1000, nonlinear_defaults["capacitor_joules"])
        z_out_amp = st.number_input("出力インピーダンス (Ω)", 0.0, 10.0, nonlinear_defaults["output_impedance"])
        z_in_amp = st.number_input("入力インピーダンス (Ω)", 1000.0, 100000.0, 47000.0)

        with st.expander("📉 小信号安定性モデル", expanded=False):
            loop_gain_db = st.slider("可聴帯ループ余裕 (dB)", 10.0, 90.0, float(small_signal_defaults["loop_gain_db"]), step=1.0)
            input_lowpass_khz = st.slider("入力帯域 (kHz)", 20, 800, int(small_signal_defaults["input_lowpass_hz"] / 1000.0), step=10)
            capacitive_sensitivity = st.slider("容量負荷感度", 0.0, 2.0, float(small_signal_defaults["capacitive_sensitivity"]), step=0.01)
            stability_margin = st.slider("安定度マージン", 0.50, 1.20, float(small_signal_defaults["stability_margin"]), step=0.01)

    with st.expander("🧵 [3] 下流ケーブル (SPK/IEM)", expanded=True):
        s_mat = st.selectbox("SPK 材質", list(CableModel.MATERIALS.keys()), key="s_mat")
        s_die = st.selectbox("SPK 被膜", list(CableModel.DIELECTRICS.keys()), key="s_die")
        s_len = st.slider("SPK 長さ (m)", 0.1, 50.0, 3.0, key="s_len")
        s_dia = st.slider("SPK 線径 (mm)", 0.01, 5.0, 2.0, step=0.01, key="s_dia")
        spk_model = CableModel(
            length=s_len,
            diameter=s_dia * 1e-3,
            spacing=5.0e-3,
            material=s_mat,
            dielectric=s_die,
            geometry="Parallel",
        )

    with st.expander("🔊 [4] 最終負荷 (Load)", expanded=False):
        load_preset = st.selectbox("負荷選択", ["Speaker (8Ω)", "IEM (20Ω)", "HP (300Ω)"])
        if load_preset == "Speaker (8Ω)":
            z_l_r = 8.0
            z_l_l = 0.5
        elif load_preset == "IEM (20Ω)":
            z_l_r = 20.0
            z_l_l = 0.01
        else:
            z_l_r = 300.0
            z_l_l = 0.1
        z_l_r = st.number_input("抵抗 (Ω)", 1.0, 1000.0, z_l_r)
        z_l_l = st.number_input("インダクタンス (mH)", 0.0, 10.0, z_l_l)

        speaker_load_model = None
        if load_preset == "Speaker (8Ω)":
            with st.expander("🧲 スピーカー負荷モデル", expanded=False):
                speaker_load_preset = st.selectbox("スピーカー負荷プリセット", list(SpeakerLoadModel.PRESETS.keys()))
                speaker_defaults = SpeakerLoadModel.PRESETS[speaker_load_preset]
                resonance_freq_hz = st.slider(
                    "低域共振周波数 (Hz)",
                    20,
                    120,
                    int(speaker_defaults["resonance_freq_hz"]),
                )
                resonance_q = st.slider(
                    "低域共振Q",
                    0.40,
                    2.00,
                    float(speaker_defaults["resonance_q"]),
                    step=0.01,
                )
                motional_peak_ohm = st.slider(
                    "逆起電力ピーク (Ω)",
                    0.0,
                    50.0,
                    float(speaker_defaults["motional_peak_ohm"]),
                    step=0.5,
                )
                crossover_freq_hz = st.slider(
                    "クロスオーバー周波数 (Hz)",
                    500,
                    5000,
                    int(speaker_defaults["crossover_freq_hz"]),
                    step=50,
                )
                back_emf_strength = st.slider(
                    "逆起電力感度",
                    0.0,
                    1.5,
                    float(speaker_defaults["back_emf_strength"]),
                    step=0.01,
                )
                current_damping_sensitivity = st.slider(
                    "駆動力低下感度",
                    0.1,
                    2.0,
                    float(speaker_defaults["current_damping_sensitivity"]),
                    step=0.01,
                )

load_inductance_h = z_l_l * 1e-3
power_load_model = None
if load_preset == "Speaker (8Ω)":
    power_load_model = SpeakerLoadModel.from_preset(
        speaker_load_preset,
        nominal_impedance=z_l_r,
        voice_coil_inductance_h=load_inductance_h,
        resonance_freq_hz=resonance_freq_hz,
        resonance_q=resonance_q,
        motional_peak_ohm=motional_peak_ohm,
        crossover_freq_hz=crossover_freq_hz,
        back_emf_strength=back_emf_strength,
        current_damping_sensitivity=current_damping_sensitivity,
    )

line_interface_model = InterfaceModel.from_preset(
    line_stage_preset,
    output_impedance=l_z_src,
    input_impedance=z_in_amp,
    source_capacitance_pf=source_cap_pf,
    input_capacitance_pf=input_cap_pf,
    stage_bandwidth_hz=stage_bw_khz * 1000.0,
    settling_strength=settling_strength,
    stereo_ground_coupling=stereo_ground_coupling,
    shield_coverage=shield_coverage,
    ingress_sensitivity=ingress_sensitivity,
    plug_contamination=plug_contamination,
    contact_nonlinearity=contact_nonlinearity,
)

amp_nonlinear_model = AmplifierModel(
    input_impedance=z_in_amp,
    output_impedance=z_out_amp,
    slew_rate=sr_val,
    capacitor_joules=cap_val,
    harmonics_2nd=h2,
    harmonics_3rd=h3,
)

amp_small_signal_model = AmplifierSmallSignalModel.from_preset(
    amp_type,
    output_impedance=z_out_amp,
    loop_gain_db=loop_gain_db,
    input_lowpass_hz=input_lowpass_khz * 1000.0,
    capacitive_sensitivity=capacitive_sensitivity,
    stability_margin=stability_margin,
)

chain = AudioSystemChain(
    line_stage=LineStageConfig(
        cable_model=line_model,
        interface_model=line_interface_model,
        source_impedance=l_z_src,
        load_impedance=z_in_amp,
    ),
    power_stage=PowerStageConfig(
        amplifier_nonlinear=amp_nonlinear_model,
        amplifier_small_signal=amp_small_signal_model,
        cable_model=spk_model,
        load_resistance=z_l_r,
        load_inductance_h=load_inductance_h,
        load_model=power_load_model,
    ),
    fir_taps=FIR_TAPS,
    analysis_ir_samples=ANALYSIS_IR_SAMPLES,
)

analysis_sr = (
    st.session_state.sample_rate
    if st.session_state.audio_source == "アップロード" and st.session_state.audio_data is not None
    else DEFAULT_SAMPLE_RATE
)
plot_max_freq = min(20000.0, analysis_sr * 0.49)
analysis = chain.analyze(sample_rate=analysis_sr, plot_max_freq=plot_max_freq)

current_preview_signature = build_preview_signature(
    st.session_state.audio_source,
    analysis_sr,
    st.session_state.audio_data if st.session_state.audio_source == "アップロード" else None,
    line_stage_preset=line_stage_preset,
    l_len=float(l_len),
    l_dia=float(l_dia),
    l_z_src=float(l_z_src),
    l_contact_res=float(l_contact_res),
    source_cap_pf=float(source_cap_pf),
    input_cap_pf=float(input_cap_pf),
    stage_bw_khz=int(stage_bw_khz),
    settling_strength=float(settling_strength),
    stereo_ground_coupling=float(stereo_ground_coupling),
    shield_coverage=float(shield_coverage),
    ingress_sensitivity=float(ingress_sensitivity),
    plug_contamination=float(plug_contamination),
    contact_nonlinearity=float(contact_nonlinearity),
    amp_type=amp_type,
    h2=float(h2),
    h3=float(h3),
    sr_val=float(sr_val),
    cap_val=int(cap_val),
    z_out_amp=float(z_out_amp),
    z_in_amp=float(z_in_amp),
    loop_gain_db=float(loop_gain_db),
    input_lowpass_khz=int(input_lowpass_khz),
    capacitive_sensitivity=float(capacitive_sensitivity),
    stability_margin=float(stability_margin),
    s_len=float(s_len),
    s_dia=float(s_dia),
    load_preset=load_preset,
    z_l_r=float(z_l_r),
    z_l_l=float(z_l_l),
    speaker_load_preset=speaker_load_preset if load_preset == "Speaker (8Ω)" else None,
)

with col_plot:
    st.header("📊 システム解析モニター")

    amp_diag = analysis["amp_diagnostics"]

    row1 = st.columns(4)
    with row1[0]:
        st.metric(
            "スピーカー側 ダンピングファクター",
            f"{analysis['damping_factor']:.1f}",
            help="制動力の指標。ケーブル直列抵抗込みで計算しています。",
        )
    with row1[1]:
        st.metric("音の質感 (TailRatio)", f"{analysis['tail_ratio_db']:.1f} dB", help="ピーク後 0.1-5ms の尾エネルギー比。")
    with row1[2]:
        st.metric("RCA 容量負荷", f"{analysis['line_capacitance_pf']:.0f} pF", help=f"現在のストレス指標: {analysis['line_stress']:.2f}")
    with row1[3]:
        st.metric("L/R 漏れ @10kHz", f"{analysis['line_crosstalk_db']:.1f} dB", help="共通リターン由来のステレオ汚染推定値。")

    row2 = st.columns(4)
    with row2[0]:
        st.metric("音場誤差 (StageError)", f"{analysis['stage_error_ms']:.3f} ms", help="群遅延の直線成分を除いたRMS誤差。")
    with row2[1]:
        st.metric("ステップ応答オーバーシュート", f"{analysis['step_metrics']['overshoot_pct']:.1f} %", help="立ち上がり後のリンギング量。")
    with row2[2]:
        st.metric(
            "アンプ位相余裕",
            f"{amp_diag['phase_margin_deg']:.1f} deg",
            help=f"容量負荷込みの概算位相余裕。ループ余裕の基準帯域は約 {amp_diag['loop_gain_anchor_hz'] / 1000.0:.1f} kHz。",
        )
    with row2[3]:
        st.metric("Zout @20kHz", f"{amp_diag['output_impedance_20khz']:.3f} Ω", help=f"負荷容量で極移動 {amp_diag['load_pole_shift_pct']:.1f}%")

    row3 = st.columns(4)
    with row3[0]:
        st.metric("駆動力ロス", f"{analysis['drive_loss_pct']:.1f} %", help="長尺ケーブルと低インピーダンス負荷で増えやすい駆動余裕の損失推定。")
    with row3[1]:
        st.metric("最小負荷インピーダンス", f"{analysis['load_min_impedance_ohm']:.2f} Ω", help="複雑負荷モデル時の最低インピーダンス。")
    with row3[2]:
        st.metric("パワー段 直列抵抗", f"{analysis['power_series_resistance_ohm']:.3f} Ω", help="アンプ出力Zとスピーカーケーブル抵抗の合計。")
    with row3[3]:
        st.metric("シールド侵入推定", f"{analysis['line_ingress_db']:.1f} dB", help=f"接点劣化指標: {analysis['line_contact_severity_pct']:.1f} %")

    fig, axes = plt.subplots(5, 1, figsize=(12, 20))

    total_response = analysis["responses"]["total"]
    axes[0].semilogx(analysis["freqs_hz"], 20 * np.log10(np.maximum(np.abs(total_response), 1e-12)), color="red", lw=2)
    axes[0].set_title("周波数特性 (Magnitude Response)")
    axes[0].set_ylabel("Gain (dB)")
    axes[0].set_xlim(20, plot_max_freq)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(analysis["freqs_hz"], np.rad2deg(analysis["phase_rad"]), color="orange")
    axes[1].set_title("位相特性 (Phase Response)")
    axes[1].set_ylabel("Phase (deg)")
    axes[1].set_xlim(20, plot_max_freq)
    axes[1].grid(True, alpha=0.3)

    gd = analysis["group_delay_us"]
    axes[2].semilogx(analysis["freqs_hz"], gd, color="purple")
    axes[2].set_title("群遅延 (Group Delay)")
    axes[2].set_ylabel("Delay (μs)")
    axes[2].set_xlim(20, plot_max_freq)
    audible = (analysis["freqs_hz"] >= 20.0) & (analysis["freqs_hz"] <= plot_max_freq)
    gd_valid = gd[audible]
    if len(gd_valid) > 0:
        axes[2].set_ylim(np.min(gd_valid) - 10, np.max(gd_valid) + 10)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(analysis["t_ir_ms"], analysis["ir_display"], color="cyan")
    axes[3].set_title("過渡応答 (Impulse Response) - システム全体")
    axes[3].set_xlim(-0.5, 2.0)
    axes[3].set_xlabel("Time (ms)")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(analysis["t_step_ms"], analysis["step_display"], color="green")
    axes[4].set_title("ステップ応答 (Step Response)")
    axes[4].set_xlim(-0.2, 3.0)
    axes[4].set_xlabel("Time (ms)")
    axes[4].set_ylabel("Normalized")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader("🎵 試聴プレイヤー")
    audio_source = st.radio(
        "音声ソース",
        ["アップロード", "サインスイープ", "ホワイトノイズ"],
        horizontal=True,
        key="audio_source",
    )

    current_data, current_sr = get_audio_source_data(audio_source)

    if current_data is None and audio_source == "アップロード":
        st.warning("⚠️ WAVファイルをアップロードしてください。")
        st.button("▶ システム全体を通して再生 (聴き比べ)", use_container_width=True, disabled=True)
    else:
        if st.button("▶ システム全体を通して再生 (聴き比べ)", use_container_width=True):
            with st.spinner("信号がチェーンを通過中..."):
                original = np.asarray(current_data, dtype=np.float64)
                processed = chain.process_audio(original, current_sr, normalize_output=False)
                difference = processed - original
                difference_rms = float(np.sqrt(np.mean(difference ** 2))) if difference.size > 0 else 0.0
                cleanup_preview_files(st.session_state.preview_payload)
                st.session_state.preview_payload = {
                    "signature": current_preview_signature,
                    "sample_rate": current_sr,
                    "difference_rms": difference_rms,
                    "original_peak": float(np.max(np.abs(original))) if original.size > 0 else 0.0,
                    "processed_peak": float(np.max(np.abs(processed))) if processed.size > 0 else 0.0,
                    "difference_peak": float(np.max(np.abs(difference))) if difference.size > 0 else 0.0,
                    "original_path": audio_to_wav_file(original, current_sr, "original"),
                    "processed_path": audio_to_wav_file(processed, current_sr, "processed"),
                    "difference_path": audio_to_wav_file(difference, current_sr, "difference"),
                }

        preview_payload = st.session_state.preview_payload
        if preview_payload is not None:
            is_stale = preview_payload["signature"] != current_preview_signature
            st.write("---")
            st.markdown("**▼ オリジナル (Original Source)**")
            st.audio(preview_payload["original_path"], format="audio/wav")

            st.markdown("**▼ システム通過後 (Processed Chain)**")
            st.audio(preview_payload["processed_path"], format="audio/wav")

            st.caption(
                f"差分RMS: {preview_payload['difference_rms']:.6f}  |  "
                f"Processed peak: {preview_payload['processed_peak']:.4f}  |  "
                f"Difference peak: {preview_payload['difference_peak']:.4f}"
            )
            st.caption("『システム通過後』も『差分音』も非正規化・未レベル合わせです。")
            if is_stale:
                st.info("現在のスライダー設定とは別のプレビューです。新しい設定で聴くにはもう一度ボタンを押してにゃ。")
            if preview_payload["processed_peak"] < 1e-4:
                st.warning("加工後のピークがかなり小さいです。設定によってはほぼ無音に聞こえることがあります。")

            st.markdown("**▼ 差分音 (Processed - Original, 非正規化)**")
            st.audio(preview_payload["difference_path"], format="audio/wav")

            with open(preview_payload["processed_path"], "rb") as processed_file:
                st.download_button(
                    "💾 加工済みファイルを保存",
                    processed_file,
                    "processed_audio.wav",
                    "audio/wav",
                    use_container_width=True,
                )
