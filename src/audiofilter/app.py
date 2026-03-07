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
MAX_UPLOADED_SECONDS = 180
PREVIEW_DIR = Path(tempfile.gettempdir()) / "audio_chain_physics_previews"
AUDIO_SOURCE_OPTIONS = ["upload", "sine_sweep", "white_noise"]
LOAD_PRESET_OPTIONS = ["speaker", "iem", "hp"]

LANGUAGE_LABELS = {
    "ja": "日本語",
    "en": "English",
}

OPTION_LABELS = {
    "ja": {
        "materials": {
            "Copper": "銅 (Copper)",
            "Silver": "銀 (Silver)",
            "Gold": "金 (Gold)",
            "Aluminum": "アルミ (Aluminum)",
            "Platinum": "白金 (Platinum)",
        },
        "dielectrics": {
            "Teflon (PTFE)": "テフロン (PTFE)",
            "Polyethylene (PE)": "ポリエチレン (PE)",
            "Polypropylene (PP)": "ポリプロピレン (PP)",
            "PVC (Vinyl)": "PVC (Vinyl)",
            "Rubber": "ゴム (Rubber)",
            "Cotton/Paper": "綿/紙 (Cotton/Paper)",
        },
        "amp_preset": {
            "カスタム": "カスタム",
            "真空管(300B風)": "真空管(300B風)",
            "ハイスピード・トランジスタ": "ハイスピード・トランジスタ",
            "現代的D級アンプ": "現代的D級アンプ",
        },
        "line_stage_preset": {
            "ニュートラル・ライン出力": "ニュートラル・ライン出力",
            "真空管プリ/ヴィンテージ": "真空管プリ/ヴィンテージ",
            "高速ソリッドステート": "高速ソリッドステート",
            "長尺RCAに敏感": "長尺RCAに敏感",
            "安物/劣化RCA": "安物/劣化RCA",
        },
        "speaker_load_preset": {
            "2-way Speaker (8Ω)": "2ウェイ・スピーカー (8Ω)",
            "Long Cable Sensitive Speaker": "長尺ケーブルに敏感なスピーカー",
            "Studio Monitor (4Ω)": "スタジオモニター (4Ω)",
        },
        "audio_source": {
            "upload": "アップロード",
            "sine_sweep": "サインスイープ",
            "white_noise": "ホワイトノイズ",
        },
        "load_preset": {
            "speaker": "Speaker (8Ω)",
            "iem": "IEM (20Ω)",
            "hp": "HP (300Ω)",
        },
    },
    "en": {
        "materials": {
            "Copper": "Copper",
            "Silver": "Silver",
            "Gold": "Gold",
            "Aluminum": "Aluminum",
            "Platinum": "Platinum",
        },
        "dielectrics": {
            "Teflon (PTFE)": "Teflon (PTFE)",
            "Polyethylene (PE)": "Polyethylene (PE)",
            "Polypropylene (PP)": "Polypropylene (PP)",
            "PVC (Vinyl)": "PVC (Vinyl)",
            "Rubber": "Rubber",
            "Cotton/Paper": "Cotton/Paper",
        },
        "amp_preset": {
            "カスタム": "Custom",
            "真空管(300B風)": "Tube-like (300B-inspired)",
            "ハイスピード・トランジスタ": "High-speed transistor",
            "現代的D級アンプ": "Modern Class-D amplifier",
        },
        "line_stage_preset": {
            "ニュートラル・ライン出力": "Neutral line output",
            "真空管プリ/ヴィンテージ": "Tube preamp / vintage",
            "高速ソリッドステート": "High-speed solid-state",
            "長尺RCAに敏感": "Sensitive to long RCA runs",
            "安物/劣化RCA": "Cheap / degraded RCA",
        },
        "speaker_load_preset": {
            "2-way Speaker (8Ω)": "2-way Speaker (8Ω)",
            "Long Cable Sensitive Speaker": "Long Cable Sensitive Speaker",
            "Studio Monitor (4Ω)": "Studio Monitor (4Ω)",
        },
        "audio_source": {
            "upload": "Upload",
            "sine_sweep": "Sine sweep",
            "white_noise": "White noise",
        },
        "load_preset": {
            "speaker": "Speaker (8Ω)",
            "iem": "IEM (20Ω)",
            "hp": "Headphones (300Ω)",
        },
    },
}

I18N = {
    "ja": {
        "page_title": "Audio System Chain Simulator v1.7",
        "language_label": "言語 / Language",
        "title": "オーディオ・システム・チェーン・シミュレーター v1.7",
        "system_config": "⚙️ システム構成",
        "section_line": "🔌 [1] 上流ケーブル (RCA)",
        "line_material": "RCA 材質",
        "line_dielectric": "RCA 被膜",
        "line_length": "RCA 長さ (m)",
        "line_diameter": "RCA 芯線径 (mm)",
        "line_source_impedance": "出力 Z (Ω)",
        "line_contact_res": "RCA 接点抵抗 (Ω)",
        "line_detail": "🧪 RCAインターフェース詳細",
        "line_preset": "出力段プリセット",
        "line_detail_caption": "長尺RCAや機器差をここで追い込めます。",
        "source_cap_pf": "出力側浮遊容量 (pF)",
        "input_cap_pf": "入力側容量 (pF)",
        "stage_bw_khz": "出力段帯域 (kHz)",
        "settling_strength": "容量負荷セトリング感度",
        "stereo_ground_coupling": "共通GND結合",
        "shield_coverage": "シールド品質",
        "ingress_sensitivity": "RF侵入感度",
        "plug_contamination": "プラグ汚れ/接点劣化",
        "contact_nonlinearity": "接点非線形",
        "section_amp": "🎛️ [2] アンプ (Amplifier)",
        "amp_preset": "アンプ・プリセット",
        "h2": "2次高調波 (温かみ)",
        "h3": "3次高調波 (輝き)",
        "slew_rate": "スルーレート (V/us)",
        "cap_val": "電源余裕 (コンデンサ)",
        "z_out_amp": "出力インピーダンス (Ω)",
        "z_in_amp": "入力インピーダンス (Ω)",
        "amp_small_signal": "📉 小信号安定性モデル",
        "loop_gain_db": "可聴帯ループ余裕 (dB)",
        "input_lowpass_khz": "入力帯域 (kHz)",
        "capacitive_sensitivity": "容量負荷感度",
        "stability_margin": "安定度マージン",
        "section_speaker_cable": "🧵 [3] 下流ケーブル (SPK/IEM)",
        "speaker_material": "SPK 材質",
        "speaker_dielectric": "SPK 被膜",
        "speaker_length": "SPK 長さ (m)",
        "speaker_diameter": "SPK 線径 (mm)",
        "section_load": "🔊 [4] 最終負荷 (Load)",
        "load_preset": "負荷選択",
        "load_resistance": "抵抗 (Ω)",
        "load_inductance": "インダクタンス (mH)",
        "speaker_load_model": "🧲 スピーカー負荷モデル",
        "speaker_load_preset": "スピーカー負荷プリセット",
        "resonance_freq_hz": "低域共振周波数 (Hz)",
        "resonance_q": "低域共振Q",
        "motional_peak_ohm": "逆起電力ピーク (Ω)",
        "crossover_freq_hz": "クロスオーバー周波数 (Hz)",
        "back_emf_strength": "逆起電力感度",
        "current_damping_sensitivity": "駆動力低下感度",
        "analysis_header": "📊 システム解析モニター",
        "metric_damping": "スピーカー側 ダンピングファクター",
        "metric_damping_help": "制動力の指標。ケーブル直列抵抗込みで計算しています。",
        "metric_tail": "音の質感 (TailRatio)",
        "metric_tail_help": "ピーク後 0.1-5ms の尾エネルギー比。",
        "metric_rca_cap": "RCA 容量負荷",
        "metric_rca_cap_help": "現在のストレス指標: {stress:.2f}",
        "metric_crosstalk": "L/R 漏れ @10kHz",
        "metric_crosstalk_help": "共通リターン由来のステレオ汚染推定値。",
        "metric_stage_error": "音場誤差 (StageError)",
        "metric_stage_error_help": "群遅延の直線成分を除いたRMS誤差。",
        "metric_overshoot": "ステップ応答オーバーシュート",
        "metric_overshoot_help": "立ち上がり後のリンギング量。",
        "metric_phase_margin": "アンプ位相余裕",
        "metric_phase_margin_help": "容量負荷込みの概算位相余裕。ループ余裕の基準帯域は約 {anchor_khz:.1f} kHz。",
        "metric_zout_20k": "Zout @20kHz",
        "metric_zout_20k_help": "負荷容量で極移動 {shift_pct:.1f}%",
        "metric_drive_loss": "駆動力ロス",
        "metric_drive_loss_help": "長尺ケーブルと低インピーダンス負荷で増えやすい駆動余裕の損失推定。",
        "metric_load_min": "最小負荷インピーダンス",
        "metric_load_min_help": "複雑負荷モデル時の最低インピーダンス。",
        "metric_series_res": "パワー段 直列抵抗",
        "metric_series_res_help": "アンプ出力Zとスピーカーケーブル抵抗の合計。",
        "metric_ingress": "シールド侵入推定",
        "metric_ingress_help": "接点劣化指標: {severity:.1f} %",
        "plot_mag": "周波数特性 (Magnitude Response)",
        "plot_phase": "位相特性 (Phase Response)",
        "plot_group_delay": "群遅延 (Group Delay)",
        "plot_impulse": "過渡応答 (Impulse Response) - システム全体",
        "plot_step": "ステップ応答 (Step Response)",
        "axis_gain_db": "ゲイン (dB)",
        "axis_phase_deg": "位相 (deg)",
        "axis_delay_us": "遅延 (μs)",
        "axis_time_ms": "時間 (ms)",
        "axis_normalized": "正規化",
        "audio_player": "🎵 試聴プレイヤー",
        "audio_source": "音声ソース",
        "upload_file": "WAVファイルをアップロード",
        "upload_notice": "アップロード音声は {seconds:.1f} 秒ありました。Cloud 安定化のため、先頭 {limit} 秒だけを使用しています。",
        "need_upload": "⚠️ WAVファイルをアップロードしてください。",
        "play_button": "▶ システム全体を通して再生 (聴き比べ)",
        "processing": "信号がチェーンを通過中...",
        "preview_original": "**▼ オリジナル (Original Source)**",
        "preview_processed": "**▼ システム通過後 (Processed Chain)**",
        "preview_caption_metrics": "差分RMS: {difference_rms:.6f}  |  Processed peak: {processed_peak:.4f}  |  Difference peak: {difference_peak:.4f}",
        "preview_caption_note": "『システム通過後』も『差分音』も非正規化・未レベル合わせです。",
        "preview_stale": "現在のスライダー設定とは別のプレビューです。新しい設定で聴くにはもう一度ボタンを押してにゃ。",
        "preview_low_peak": "加工後のピークがかなり小さいです。設定によってはほぼ無音に聞こえることがあります。",
        "preview_difference": "**▼ 差分音 (Processed - Original, 非正規化)**",
        "download_processed": "💾 加工済みファイルを保存",
        "download_filename": "processed_audio_ja.wav",
    },
    "en": {
        "page_title": "Audio System Chain Simulator v1.7",
        "language_label": "Language / 言語",
        "title": "Audio System Chain Simulator v1.7",
        "system_config": "⚙️ System Configuration",
        "section_line": "🔌 [1] Upstream Cable (RCA)",
        "line_material": "RCA conductor material",
        "line_dielectric": "RCA dielectric",
        "line_length": "RCA length (m)",
        "line_diameter": "RCA conductor diameter (mm)",
        "line_source_impedance": "Source impedance (Ω)",
        "line_contact_res": "RCA contact resistance (Ω)",
        "line_detail": "🧪 RCA Interface Details",
        "line_preset": "Line-stage preset",
        "line_detail_caption": "Use this section to explore long RCA runs and source/device interaction.",
        "source_cap_pf": "Source-side stray capacitance (pF)",
        "input_cap_pf": "Input capacitance (pF)",
        "stage_bw_khz": "Output-stage bandwidth (kHz)",
        "settling_strength": "Capacitive settling sensitivity",
        "stereo_ground_coupling": "Shared ground coupling",
        "shield_coverage": "Shield quality",
        "ingress_sensitivity": "RF ingress sensitivity",
        "plug_contamination": "Plug contamination / contact aging",
        "contact_nonlinearity": "Contact nonlinearity",
        "section_amp": "🎛️ [2] Amplifier",
        "amp_preset": "Amplifier preset",
        "h2": "2nd harmonic (warmth)",
        "h3": "3rd harmonic (shine)",
        "slew_rate": "Slew rate (V/us)",
        "cap_val": "Power reserve (capacitor)",
        "z_out_amp": "Output impedance (Ω)",
        "z_in_amp": "Input impedance (Ω)",
        "amp_small_signal": "📉 Small-Signal Stability Model",
        "loop_gain_db": "Audible-band loop headroom (dB)",
        "input_lowpass_khz": "Input bandwidth (kHz)",
        "capacitive_sensitivity": "Capacitive load sensitivity",
        "stability_margin": "Stability margin",
        "section_speaker_cable": "🧵 [3] Downstream Cable (SPK/IEM)",
        "speaker_material": "SPK conductor material",
        "speaker_dielectric": "SPK dielectric",
        "speaker_length": "SPK length (m)",
        "speaker_diameter": "SPK conductor diameter (mm)",
        "section_load": "🔊 [4] Final Load",
        "load_preset": "Load preset",
        "load_resistance": "Resistance (Ω)",
        "load_inductance": "Inductance (mH)",
        "speaker_load_model": "🧲 Speaker Load Model",
        "speaker_load_preset": "Speaker load preset",
        "resonance_freq_hz": "Low-frequency resonance (Hz)",
        "resonance_q": "Low-frequency resonance Q",
        "motional_peak_ohm": "Back-EMF peak (Ω)",
        "crossover_freq_hz": "Crossover frequency (Hz)",
        "back_emf_strength": "Back-EMF strength",
        "current_damping_sensitivity": "Drive-loss sensitivity",
        "analysis_header": "📊 System Analysis Monitor",
        "metric_damping": "Speaker-side damping factor",
        "metric_damping_help": "A control/damping indicator computed including cable series resistance.",
        "metric_tail": "Texture metric (TailRatio)",
        "metric_tail_help": "Tail energy ratio in the 0.1-5 ms window after the peak.",
        "metric_rca_cap": "RCA capacitive load",
        "metric_rca_cap_help": "Current stress indicator: {stress:.2f}",
        "metric_crosstalk": "L/R leakage @10kHz",
        "metric_crosstalk_help": "Estimated stereo contamination caused by the shared return path.",
        "metric_stage_error": "Stage error (StageError)",
        "metric_stage_error_help": "RMS group-delay error after removing the linear component.",
        "metric_overshoot": "Step-response overshoot",
        "metric_overshoot_help": "Amount of ringing after the initial rise.",
        "metric_phase_margin": "Amplifier phase margin",
        "metric_phase_margin_help": "Approximate phase margin including capacitive loading. Loop headroom anchor is about {anchor_khz:.1f} kHz.",
        "metric_zout_20k": "Zout @20kHz",
        "metric_zout_20k_help": "Estimated pole shift from load capacitance: {shift_pct:.1f}%",
        "metric_drive_loss": "Drive loss",
        "metric_drive_loss_help": "Estimated loss of drive reserve that grows with long cables and low-impedance loads.",
        "metric_load_min": "Minimum load impedance",
        "metric_load_min_help": "Minimum impedance under the complex-load model.",
        "metric_series_res": "Power-stage series resistance",
        "metric_series_res_help": "Combined amplifier output impedance and speaker-cable resistance.",
        "metric_ingress": "Shield ingress estimate",
        "metric_ingress_help": "Contact degradation indicator: {severity:.1f} %",
        "plot_mag": "Magnitude Response",
        "plot_phase": "Phase Response",
        "plot_group_delay": "Group Delay",
        "plot_impulse": "Impulse Response - Full System",
        "plot_step": "Step Response",
        "axis_gain_db": "Gain (dB)",
        "axis_phase_deg": "Phase (deg)",
        "axis_delay_us": "Delay (μs)",
        "axis_time_ms": "Time (ms)",
        "axis_normalized": "Normalized",
        "audio_player": "🎵 Listening Preview",
        "audio_source": "Audio source",
        "upload_file": "Upload a WAV file",
        "upload_notice": "The uploaded file was {seconds:.1f} seconds long. For Cloud stability, only the first {limit} seconds are used.",
        "need_upload": "⚠️ Please upload a WAV file.",
        "play_button": "▶ Process Through Full Chain (A/B Preview)",
        "processing": "Processing through the chain...",
        "preview_original": "**▼ Original Source**",
        "preview_processed": "**▼ Processed Chain**",
        "preview_caption_metrics": "Difference RMS: {difference_rms:.6f}  |  Processed peak: {processed_peak:.4f}  |  Difference peak: {difference_peak:.4f}",
        "preview_caption_note": "Both the processed output and the difference signal are unnormalized and not loudness-matched.",
        "preview_stale": "This preview was generated with a different slider state. Press the button again to hear the current settings.",
        "preview_low_peak": "The processed signal peak is very small. Depending on the current settings, it may sound almost silent.",
        "preview_difference": "**▼ Difference Signal (Processed - Original, unnormalized)**",
        "download_processed": "💾 Download processed audio",
        "download_filename": "processed_audio_en.wav",
    },
}

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
            cleanup_preview_files(st.session_state.get("preview_payload"))
            st.session_state.preview_payload = None
            st.session_state.audio_data = None
            st.session_state.sample_rate = DEFAULT_SAMPLE_RATE
            st.session_state.uploaded_audio_signature = None
            st.session_state.upload_notice = None
        return

    signature = (uploaded_file.name, uploaded_file.size)
    if signature == st.session_state.get("uploaded_audio_signature"):
        return

    uploaded_file.seek(0)
    with sf.SoundFile(uploaded_file) as audio_file:
        sr = int(audio_file.samplerate)
        total_frames = int(len(audio_file))
        max_frames = int(MAX_UPLOADED_SECONDS * sr)
        frames_to_read = min(total_frames, max_frames)
        data = audio_file.read(frames=frames_to_read, dtype="float64", always_2d=False)

    cleanup_preview_files(st.session_state.get("preview_payload"))
    st.session_state.preview_payload = None
    st.session_state.audio_data = data
    st.session_state.sample_rate = sr
    st.session_state.uploaded_audio_signature = signature
    if total_frames > max_frames:
        st.session_state.upload_notice = total_frames / sr
    else:
        st.session_state.upload_notice = None


def get_audio_source_data(audio_source):
    if audio_source == "upload":
        st.file_uploader(tr("upload_file"), type=["wav"], key="uploaded_wav")
        upload_notice = st.session_state.get("upload_notice")
        if upload_notice is not None:
            st.info(tr("upload_notice", seconds=upload_notice, limit=MAX_UPLOADED_SECONDS))
        current_data = st.session_state.audio_data
        current_sr = st.session_state.sample_rate if current_data is not None else DEFAULT_SAMPLE_RATE
        return current_data, current_sr

    if audio_source == "white_noise":
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


def preview_files_exist(payload):
    if not payload:
        return False
    return all(Path(payload.get(key, "")).exists() for key in ("original_path", "processed_path", "difference_path"))


def build_preview_signature(audio_source, current_sr, current_data, **settings):
    data_shape = tuple(current_data.shape) if current_data is not None else None
    return (
        audio_source,
        current_sr,
        data_shape,
        tuple(sorted(settings.items())),
        st.session_state.get("uploaded_audio_signature"),
    )


def get_language():
    return st.session_state.get("language", "ja")


def tr(key, **kwargs):
    return I18N[get_language()][key].format(**kwargs)


def option_label(group, value):
    return OPTION_LABELS[get_language()][group].get(value, value)


st.set_page_config(page_title=I18N["en"]["page_title"], layout="wide")

if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "sample_rate" not in st.session_state:
    st.session_state.sample_rate = DEFAULT_SAMPLE_RATE
if "language" not in st.session_state:
    st.session_state.language = "ja"
if "audio_source" not in st.session_state:
    st.session_state.audio_source = "upload"
if "uploaded_audio_signature" not in st.session_state:
    st.session_state.uploaded_audio_signature = None
if "upload_notice" not in st.session_state:
    st.session_state.upload_notice = None
if "preview_payload" not in st.session_state:
    st.session_state.preview_payload = None

sync_uploaded_audio()

header_col_title, header_col_lang = st.columns([5, 1])
with header_col_lang:
    st.selectbox(
        tr("language_label"),
        ["ja", "en"],
        key="language",
        format_func=lambda value: LANGUAGE_LABELS[value],
    )
with header_col_title:
    st.title(tr("title"))

col_config, col_plot = st.columns([1, 2])

with col_config:
    st.header(tr("system_config"))

    with st.expander(tr("section_line"), expanded=True):
        l_mat = st.selectbox(
            tr("line_material"),
            list(CableModel.MATERIALS.keys()),
            key="l_mat",
            format_func=lambda value: option_label("materials", value),
        )
        l_die = st.selectbox(
            tr("line_dielectric"),
            list(CableModel.DIELECTRICS.keys()),
            key="l_die",
            format_func=lambda value: option_label("dielectrics", value),
        )
        l_len = st.slider(tr("line_length"), 0.1, 10.0, 1.0, key="l_len")
        l_dia = st.slider(tr("line_diameter"), 0.01, 1.0, 0.3, step=0.01, key="l_dia")
        l_z_src = st.number_input(tr("line_source_impedance"), 0.1, 2000.0, 100.0, key="l_zs")
        l_contact_res = st.number_input(tr("line_contact_res"), 0.0, 0.2, 0.01, step=0.001, format="%.3f")

        line_model = CableModel(
            length=l_len,
            diameter=l_dia * 1e-3,
            spacing=1.5e-3,
            material=l_mat,
            dielectric=l_die,
            geometry="Coaxial",
            contact_res=l_contact_res,
        )

        with st.expander(tr("line_detail"), expanded=False):
            line_stage_preset = st.selectbox(
                tr("line_preset"),
                list(InterfaceModel.PRESETS.keys()),
                key="line_stage_preset",
                format_func=lambda value: option_label("line_stage_preset", value),
            )
            line_stage_defaults = InterfaceModel.PRESETS[line_stage_preset]
            st.caption(tr("line_detail_caption"))
            source_cap_pf = st.slider(tr("source_cap_pf"), 10.0, 400.0, float(line_stage_defaults["source_capacitance_pf"]), step=5.0)
            input_cap_pf = st.slider(tr("input_cap_pf"), 20.0, 600.0, float(line_stage_defaults["input_capacitance_pf"]), step=10.0)
            stage_bw_khz = st.slider(tr("stage_bw_khz"), 50, 1000, int(line_stage_defaults["stage_bandwidth_hz"] / 1000.0), step=10)
            settling_strength = st.slider(tr("settling_strength"), 0.0, 1.0, float(line_stage_defaults["settling_strength"]), step=0.01)
            stereo_ground_coupling = st.slider(tr("stereo_ground_coupling"), 0.0, 1.0, float(line_stage_defaults["stereo_ground_coupling"]), step=0.01)
            shield_coverage = st.slider(tr("shield_coverage"), 0.0, 1.0, float(line_stage_defaults["shield_coverage"]), step=0.01)
            ingress_sensitivity = st.slider(tr("ingress_sensitivity"), 0.0, 1.5, float(line_stage_defaults["ingress_sensitivity"]), step=0.01)
            plug_contamination = st.slider(tr("plug_contamination"), 0.0, 1.0, float(line_stage_defaults["plug_contamination"]), step=0.01)
            contact_nonlinearity = st.slider(tr("contact_nonlinearity"), 0.0, 1.0, float(line_stage_defaults["contact_nonlinearity"]), step=0.01)

    with st.expander(tr("section_amp"), expanded=True):
        amp_type = st.selectbox(
            tr("amp_preset"),
            list(NONLINEAR_AMP_PRESETS.keys()),
            format_func=lambda value: option_label("amp_preset", value),
        )
        nonlinear_defaults = NONLINEAR_AMP_PRESETS[amp_type]
        small_signal_defaults = AmplifierSmallSignalModel.PRESETS.get(amp_type, AmplifierSmallSignalModel.PRESETS["カスタム"])

        h2 = st.slider(tr("h2"), 0.0, 0.2, nonlinear_defaults["h2"], step=0.001)
        h3 = st.slider(tr("h3"), 0.0, 0.2, nonlinear_defaults["h3"], step=0.001)
        sr_val = st.slider(tr("slew_rate"), 1.0, 200.0, nonlinear_defaults["slew_rate"])
        cap_val = st.slider(tr("cap_val"), 1, 1000, nonlinear_defaults["capacitor_joules"])
        z_out_amp = st.number_input(tr("z_out_amp"), 0.0, 10.0, nonlinear_defaults["output_impedance"])
        z_in_amp = st.number_input(tr("z_in_amp"), 1000.0, 100000.0, 47000.0)

        with st.expander(tr("amp_small_signal"), expanded=False):
            loop_gain_db = st.slider(tr("loop_gain_db"), 10.0, 90.0, float(small_signal_defaults["loop_gain_db"]), step=1.0)
            input_lowpass_khz = st.slider(tr("input_lowpass_khz"), 20, 800, int(small_signal_defaults["input_lowpass_hz"] / 1000.0), step=10)
            capacitive_sensitivity = st.slider(tr("capacitive_sensitivity"), 0.0, 2.0, float(small_signal_defaults["capacitive_sensitivity"]), step=0.01)
            stability_margin = st.slider(tr("stability_margin"), 0.50, 1.20, float(small_signal_defaults["stability_margin"]), step=0.01)

    with st.expander(tr("section_speaker_cable"), expanded=True):
        s_mat = st.selectbox(
            tr("speaker_material"),
            list(CableModel.MATERIALS.keys()),
            key="s_mat",
            format_func=lambda value: option_label("materials", value),
        )
        s_die = st.selectbox(
            tr("speaker_dielectric"),
            list(CableModel.DIELECTRICS.keys()),
            key="s_die",
            format_func=lambda value: option_label("dielectrics", value),
        )
        s_len = st.slider(tr("speaker_length"), 0.1, 50.0, 3.0, key="s_len")
        s_dia = st.slider(tr("speaker_diameter"), 0.01, 5.0, 2.0, step=0.01, key="s_dia")
        spk_model = CableModel(
            length=s_len,
            diameter=s_dia * 1e-3,
            spacing=5.0e-3,
            material=s_mat,
            dielectric=s_die,
            geometry="Parallel",
        )

    with st.expander(tr("section_load"), expanded=False):
        load_preset = st.selectbox(
            tr("load_preset"),
            LOAD_PRESET_OPTIONS,
            format_func=lambda value: option_label("load_preset", value),
        )
        if load_preset == "speaker":
            z_l_r = 8.0
            z_l_l = 0.5
        elif load_preset == "iem":
            z_l_r = 20.0
            z_l_l = 0.01
        else:
            z_l_r = 300.0
            z_l_l = 0.1
        z_l_r = st.number_input(tr("load_resistance"), 1.0, 1000.0, z_l_r)
        z_l_l = st.number_input(tr("load_inductance"), 0.0, 10.0, z_l_l)

        speaker_load_model = None
        if load_preset == "speaker":
            with st.expander(tr("speaker_load_model"), expanded=False):
                speaker_load_preset = st.selectbox(
                    tr("speaker_load_preset"),
                    list(SpeakerLoadModel.PRESETS.keys()),
                    format_func=lambda value: option_label("speaker_load_preset", value),
                )
                speaker_defaults = SpeakerLoadModel.PRESETS[speaker_load_preset]
                resonance_freq_hz = st.slider(
                    tr("resonance_freq_hz"),
                    20,
                    120,
                    int(speaker_defaults["resonance_freq_hz"]),
                )
                resonance_q = st.slider(
                    tr("resonance_q"),
                    0.40,
                    2.00,
                    float(speaker_defaults["resonance_q"]),
                    step=0.01,
                )
                motional_peak_ohm = st.slider(
                    tr("motional_peak_ohm"),
                    0.0,
                    50.0,
                    float(speaker_defaults["motional_peak_ohm"]),
                    step=0.5,
                )
                crossover_freq_hz = st.slider(
                    tr("crossover_freq_hz"),
                    500,
                    5000,
                    int(speaker_defaults["crossover_freq_hz"]),
                    step=50,
                )
                back_emf_strength = st.slider(
                    tr("back_emf_strength"),
                    0.0,
                    1.5,
                    float(speaker_defaults["back_emf_strength"]),
                    step=0.01,
                )
                current_damping_sensitivity = st.slider(
                    tr("current_damping_sensitivity"),
                    0.1,
                    2.0,
                    float(speaker_defaults["current_damping_sensitivity"]),
                    step=0.01,
                )

load_inductance_h = z_l_l * 1e-3
power_load_model = None
if load_preset == "speaker":
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
    if st.session_state.audio_source == "upload" and st.session_state.audio_data is not None
    else DEFAULT_SAMPLE_RATE
)
plot_max_freq = min(20000.0, analysis_sr * 0.49)
analysis = chain.analyze(sample_rate=analysis_sr, plot_max_freq=plot_max_freq)

current_preview_signature = build_preview_signature(
    st.session_state.audio_source,
    analysis_sr,
    st.session_state.audio_data if st.session_state.audio_source == "upload" else None,
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
    speaker_load_preset=speaker_load_preset if load_preset == "speaker" else None,
)

with col_plot:
    st.header(tr("analysis_header"))

    amp_diag = analysis["amp_diagnostics"]

    row1 = st.columns(4)
    with row1[0]:
        st.metric(
            tr("metric_damping"),
            f"{analysis['damping_factor']:.1f}",
            help=tr("metric_damping_help"),
        )
    with row1[1]:
        st.metric(tr("metric_tail"), f"{analysis['tail_ratio_db']:.1f} dB", help=tr("metric_tail_help"))
    with row1[2]:
        st.metric(
            tr("metric_rca_cap"),
            f"{analysis['line_capacitance_pf']:.0f} pF",
            help=tr("metric_rca_cap_help", stress=analysis["line_stress"]),
        )
    with row1[3]:
        st.metric(tr("metric_crosstalk"), f"{analysis['line_crosstalk_db']:.1f} dB", help=tr("metric_crosstalk_help"))

    row2 = st.columns(4)
    with row2[0]:
        st.metric(tr("metric_stage_error"), f"{analysis['stage_error_ms']:.3f} ms", help=tr("metric_stage_error_help"))
    with row2[1]:
        st.metric(tr("metric_overshoot"), f"{analysis['step_metrics']['overshoot_pct']:.1f} %", help=tr("metric_overshoot_help"))
    with row2[2]:
        st.metric(
            tr("metric_phase_margin"),
            f"{amp_diag['phase_margin_deg']:.1f} deg",
            help=tr("metric_phase_margin_help", anchor_khz=amp_diag["loop_gain_anchor_hz"] / 1000.0),
        )
    with row2[3]:
        st.metric(
            tr("metric_zout_20k"),
            f"{amp_diag['output_impedance_20khz']:.3f} Ω",
            help=tr("metric_zout_20k_help", shift_pct=amp_diag["load_pole_shift_pct"]),
        )

    row3 = st.columns(4)
    with row3[0]:
        st.metric(tr("metric_drive_loss"), f"{analysis['drive_loss_pct']:.1f} %", help=tr("metric_drive_loss_help"))
    with row3[1]:
        st.metric(tr("metric_load_min"), f"{analysis['load_min_impedance_ohm']:.2f} Ω", help=tr("metric_load_min_help"))
    with row3[2]:
        st.metric(tr("metric_series_res"), f"{analysis['power_series_resistance_ohm']:.3f} Ω", help=tr("metric_series_res_help"))
    with row3[3]:
        st.metric(
            tr("metric_ingress"),
            f"{analysis['line_ingress_db']:.1f} dB",
            help=tr("metric_ingress_help", severity=analysis["line_contact_severity_pct"]),
        )

    fig, axes = plt.subplots(5, 1, figsize=(12, 20))

    total_response = analysis["responses"]["total"]
    axes[0].semilogx(analysis["freqs_hz"], 20 * np.log10(np.maximum(np.abs(total_response), 1e-12)), color="red", lw=2)
    axes[0].set_title(tr("plot_mag"))
    axes[0].set_ylabel(tr("axis_gain_db"))
    axes[0].set_xlim(20, plot_max_freq)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(analysis["freqs_hz"], np.rad2deg(analysis["phase_rad"]), color="orange")
    axes[1].set_title(tr("plot_phase"))
    axes[1].set_ylabel(tr("axis_phase_deg"))
    axes[1].set_xlim(20, plot_max_freq)
    axes[1].grid(True, alpha=0.3)

    gd = analysis["group_delay_us"]
    axes[2].semilogx(analysis["freqs_hz"], gd, color="purple")
    axes[2].set_title(tr("plot_group_delay"))
    axes[2].set_ylabel(tr("axis_delay_us"))
    axes[2].set_xlim(20, plot_max_freq)
    audible = (analysis["freqs_hz"] >= 20.0) & (analysis["freqs_hz"] <= plot_max_freq)
    gd_valid = gd[audible]
    if len(gd_valid) > 0:
        axes[2].set_ylim(np.min(gd_valid) - 10, np.max(gd_valid) + 10)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(analysis["t_ir_ms"], analysis["ir_display"], color="cyan")
    axes[3].set_title(tr("plot_impulse"))
    axes[3].set_xlim(-0.5, 2.0)
    axes[3].set_xlabel(tr("axis_time_ms"))
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(analysis["t_step_ms"], analysis["step_display"], color="green")
    axes[4].set_title(tr("plot_step"))
    axes[4].set_xlim(-0.2, 3.0)
    axes[4].set_xlabel(tr("axis_time_ms"))
    axes[4].set_ylabel(tr("axis_normalized"))
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.subheader(tr("audio_player"))
    audio_source = st.radio(
        tr("audio_source"),
        AUDIO_SOURCE_OPTIONS,
        horizontal=True,
        key="audio_source",
        format_func=lambda value: option_label("audio_source", value),
    )

    current_data, current_sr = get_audio_source_data(audio_source)

    if current_data is None and audio_source == "upload":
        st.warning(tr("need_upload"))
        st.button(tr("play_button"), use_container_width=True, disabled=True)
    else:
        if st.button(tr("play_button"), use_container_width=True):
            with st.spinner(tr("processing")):
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
        if preview_payload is not None and not preview_files_exist(preview_payload):
            st.session_state.preview_payload = None
            preview_payload = None
        if preview_payload is not None:
            is_stale = preview_payload["signature"] != current_preview_signature
            st.write("---")
            st.markdown(tr("preview_original"))
            st.audio(preview_payload["original_path"], format="audio/wav")

            st.markdown(tr("preview_processed"))
            st.audio(preview_payload["processed_path"], format="audio/wav")

            st.caption(
                tr(
                    "preview_caption_metrics",
                    difference_rms=preview_payload["difference_rms"],
                    processed_peak=preview_payload["processed_peak"],
                    difference_peak=preview_payload["difference_peak"],
                )
            )
            st.caption(tr("preview_caption_note"))
            if is_stale:
                st.info(tr("preview_stale"))
            if preview_payload["processed_peak"] < 1e-4:
                st.warning(tr("preview_low_peak"))

            st.markdown(tr("preview_difference"))
            st.audio(preview_payload["difference_path"], format="audio/wav")

            with open(preview_payload["processed_path"], "rb") as processed_file:
                st.download_button(
                    tr("download_processed"),
                    processed_file,
                    tr("download_filename"),
                    "audio/wav",
                    use_container_width=True,
                )
