
from __future__ import annotations

import os
import tempfile

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from audio_io import (
    load_audio_from_file,
    record_audio_from_mic,
    TARGET_SR
)

from dsp_core import spectral_subtraction_filter

from whisper_ai import (
    load_whisper_model,
    transcribe_speech
)


# =========================================================
# 1. CẤU HÌNH TRANG WEB
# =========================================================

st.set_page_config(
    page_title="DSP Speech-to-Text",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CSS - GIAO DIỆN WEBSITE
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
}

.stApp {
    background-color: #f8fafc;
}

.main .block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background-color: #f1f5f9;
    border-right: 1px solid #e2e8f0;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] h2 {
    color: #1e3a5f;
    font-weight: 700;
}

[data-testid="stSidebar"] label {
    color: #334155;
    font-weight: 500;
}

/* HEADER */

.hero {
    background: linear-gradient(120deg, #eaf2fc, #f1f5f9);
    border: 1px solid #dbeafe;
    border-left: 5px solid #3979b9;
    border-radius: 12px;
    padding: 25px 30px;
    margin-bottom: 25px;
}

.hero-title {
    color: #1e3a5f;
    font-size: clamp(22px, 2.5vw, 30px);
    font-weight: 800;
    line-height: 1.5;
    margin-bottom: 8px;
}

.hero-description {
    color: #64748b;
    font-size: 13px;
    line-height: 1.7;
    margin: 0;
}

/* TIÊU ĐỀ */

h2, h3 {
    color: #1e3a5f !important;
    font-weight: 700 !important;
}

.section-caption {
    color: #718096;
    font-size: 13px;
    margin-top: -8px;
    margin-bottom: 15px;
}

/* TAB */

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid #e2e8f0;
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    font-weight: 600;
    color: #64748b;
}

.stTabs [aria-selected="true"] {
    color: #2563eb !important;
}

/* BUTTON */

.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    min-height: 42px;
    border: 1px solid #cbd5e1;
    transition: all 0.2s ease;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #2563eb, #3979b9);
    color: white;
    border: none;
}

.stButton > button:hover {
    border-color: #2563eb;
    color: #2563eb;
}

.stButton > button[kind="primary"]:hover {
    color: white;
    background: #1d4ed8;
}

/* UPLOAD */

[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px dashed #94a3b8;
    border-radius: 10px;
    padding: 15px;
}

/* INPUT */

[data-baseweb="select"] > div {
    border-radius: 8px;
}

.stTextArea textarea {
    border-radius: 9px;
    border: 1px solid #cbd5e1;
    font-size: 14px;
    line-height: 1.7;
}

/* AUDIO */

audio {
    width: 100%;
    border-radius: 8px;
}

/* ALERT */

[data-testid="stAlert"] {
    border-radius: 9px;
}

/* DIVIDER */

hr {
    border-color: #e2e8f0;
    margin-top: 25px;
    margin-bottom: 25px;
}

/* EXPANDER */

[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 9px;
    background: #ffffff;
}

/* METRIC */

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 12px;
    border-radius: 9px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        🎙️ Hệ thống Nhận dạng Giọng nói kết hợp Lọc nhiễu DSP
    </div>

    <p class="hero-description">
        BÀI TẬP LỚN XỬ LÝ TÍN HIỆU SỐ
        &nbsp; | &nbsp;
        Chuẩn hóa 16kHz
        &nbsp; | &nbsp;
        Spectral Subtraction
        &nbsp; | &nbsp;
        OpenAI Whisper
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 4. CACHE MÔ HÌNH WHISPER
# =========================================================

@st.cache_resource
def get_cached_model(model_size: str):
    return load_whisper_model(model_size)


# =========================================================
# 5. SESSION STATE
# =========================================================

if "raw_audio" not in st.session_state:
    st.session_state.raw_audio = None

if "transcription_result" not in st.session_state:
    st.session_state.transcription_result = None

if "transcription_source_used" not in st.session_state:
    st.session_state.transcription_source_used = None

if "audio_name" not in st.session_state:
    st.session_state.audio_name = None


# =========================================================
# 6. SIDEBAR - CẤU HÌNH HỆ THỐNG
# =========================================================

with st.sidebar:

    st.header("⚙️ Cấu hình Hệ thống")

    st.caption("Thiết lập mô hình nhận dạng giọng nói")

    model_size = st.selectbox(
        "Chọn mô hình Whisper:",
        options=["tiny", "base", "small", "medium"],
        index=2,
        help="Small cân bằng tốc độ và độ chính xác. Medium cần nhiều RAM hơn."
    )

    language = st.selectbox(
        "Ngôn ngữ nói:",
        options=[
            ("Tiếng Việt", "vi"),
            ("Tiếng Anh", "en")
        ],
        format_func=lambda x: x[0]
    )[1]

    st.markdown("---")

    st.caption("⚡ Cấu hình xử lý tín hiệu")

    apply_dsp = st.checkbox(
        "Bật bộ lọc nhiễu DSP (Trừ phổ)",
        value=True
    )

    transcription_source = st.selectbox(
        "Âm thanh dùng để nhận diện:",
        options=[
            "Tín hiệu gốc",
            "Tín hiệu sau lọc DSP"
        ],
        index=0,
        help="Chọn tín hiệu đầu vào cho Whisper để so sánh độ chính xác."
    )

    st.markdown("---")

    st.caption("🤖 Trạng thái mô hình AI")

    try:
        with st.spinner(f"Đang nạp Whisper ({model_size})..."):
            get_cached_model(model_size)

        st.success(f"Mô hình '{model_size}' đã sẵn sàng!")

    except Exception as e:
        st.error(f"Không thể tải mô hình Whisper: {e}")

    st.markdown("---")

    st.caption("Thông số hệ thống")

    st.caption(f"• Sampling Rate: {TARGET_SR / 1000:g} kHz")
    st.caption("• Audio Channel: Mono")
    st.caption("• Noise Reduction: Spectral Subtraction")
    st.caption("• Speech Recognition: OpenAI Whisper")


# =========================================================
# 7. CHỌN NGUỒN ÂM THANH
# =========================================================

st.markdown(
    '<div class="section-caption">CHỌN NGUỒN ÂM THANH ĐẦU VÀO</div>',
    unsafe_allow_html=True
)

tab_upload, tab_mic = st.tabs([
    "📂 Tải file âm thanh (.wav, .mp3)",
    "🎤 Thu âm trực tiếp từ Micro"
])


# ---------------------------------------------------------
# TAB 1: UPLOAD FILE
# ---------------------------------------------------------

with tab_upload:

    st.subheader("📁 Tải file âm thanh")

    st.caption(
        "Chọn file âm thanh từ máy tính để đưa vào hệ thống xử lý."
    )

    uploaded_file = st.file_uploader(
        "Chọn file âm thanh từ máy tính",
        type=["wav", "mp3", "m4a", "flac"],
        key="audio_uploader"
    )

    if uploaded_file is not None:

        if st.button(
            "📥 Đọc và Chuẩn hóa File",
            key="btn_load_file",
            use_container_width=True
        ):

            tmp_path = None

            try:

                suffix = os.path.splitext(uploaded_file.name)[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as tmp_file:

                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name

                with st.spinner(
                    "Đang chuẩn hóa âm thanh về 16kHz Mono..."
                ):

                    audio_data, _ = load_audio_from_file(tmp_path)

                    audio_data = np.asarray(
                        audio_data,
                        dtype=np.float32
                    )

                    if audio_data.size == 0:
                        raise ValueError("File không chứa dữ liệu âm thanh.")

                    st.session_state.raw_audio = audio_data
                    st.session_state.audio_name = uploaded_file.name
                    st.session_state.transcription_result = None

                st.success(
                    "✅ Đã tải và chuẩn hóa âm thanh thành công!"
                )

            except Exception as e:

                st.error(f"Lỗi khi đọc file âm thanh: {e}")

            finally:

                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)


# ---------------------------------------------------------
# TAB 2: RECORD MICROPHONE
# ---------------------------------------------------------

with tab_mic:

    st.subheader("🎙️ Thu âm trực tiếp")

    st.caption(
        "Sử dụng Micro để ghi âm giọng nói trực tiếp."
    )

    duration = st.slider(
        "Thời lượng thu âm (giây):",
        min_value=2,
        max_value=120,
        value=10,
        step=1
    )

    if st.button(
        "🔴 Bắt đầu Thu âm",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                f"🎙️ Đang thu âm trong {duration} giây... Hãy nói vào Micro!"
            ):

                audio_data, _ = record_audio_from_mic(
                    duration_sec=duration
                )

                audio_data = np.asarray(
                    audio_data,
                    dtype=np.float32
                )

                if audio_data.size == 0:
                    raise ValueError("Không thu được dữ liệu âm thanh.")

                st.session_state.raw_audio = audio_data
                st.session_state.audio_name = "Ghi âm từ Micro"
                st.session_state.transcription_result = None

            st.success("✅ Đã thu âm xong!")

        except Exception as e:

            st.error(f"Lỗi khi thu âm: {e}")


# =========================================================
# 8. XỬ LÝ DSP VÀ HIỂN THỊ TÍN HIỆU
# =========================================================

if st.session_state.raw_audio is not None:

    st.divider()

    raw_audio = st.session_state.raw_audio

    st.header("📊 Phân tích và xử lý tín hiệu")

    st.caption(
        "So sánh tín hiệu âm thanh trước và sau khi áp dụng bộ lọc nhiễu DSP."
    )

    # -----------------------------------------------------
    # XỬ LÝ DSP
    # -----------------------------------------------------

    try:

        if apply_dsp:

            with st.spinner("Đang thực hiện lọc nhiễu DSP..."):

                processed_audio = spectral_subtraction_filter(
                    raw_audio,
                    sr=TARGET_SR
                )

            processed_audio = np.asarray(
                processed_audio,
                dtype=np.float32
            )

        else:

            processed_audio = raw_audio.copy()

    except Exception as e:

        st.error(f"Lỗi trong quá trình lọc nhiễu DSP: {e}")
        st.stop()

    # -----------------------------------------------------
    # THÔNG TIN ÂM THANH
    # -----------------------------------------------------

    duration_audio = len(raw_audio) / TARGET_SR

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Tần số lấy mẫu",
        f"{TARGET_SR / 1000:g} kHz"
    )

    metric2.metric(
        "Thời lượng âm thanh",
        f"{duration_audio:.2f} giây"
    )

    metric3.metric(
        "Số lượng mẫu",
        f"{len(raw_audio):,}"
    )

    st.caption(
        f"📁 Nguồn âm thanh: {st.session_state.audio_name or 'Âm thanh đầu vào'}"
    )

    st.markdown("---")

    # -----------------------------------------------------
    # SO SÁNH TÍN HIỆU
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    time_axis = np.arange(len(raw_audio)) / TARGET_SR

    # ===== TÍN HIỆU GỐC =====

    with col1:

        st.subheader("1. Tín hiệu Gốc")

        st.caption("Original Audio - Sau chuẩn hóa 16kHz")

        st.audio(
            raw_audio,
            sample_rate=TARGET_SR
        )

        fig1, ax1 = plt.subplots(figsize=(7, 2.5))

        ax1.plot(
            time_axis,
            raw_audio,
            color="#3979b9",
            linewidth=0.65
        )

        ax1.set_ylim([-1.05, 1.05])

        ax1.set_xlabel("Thời gian (s)", fontsize=9)
        ax1.set_ylabel("Biên độ", fontsize=9)

        ax1.set_title(
            "Original Audio Signal",
            fontsize=11,
            fontweight="bold"
        )

        ax1.grid(
            True,
            linestyle="--",
            alpha=0.25
        )

        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        fig1.tight_layout()

        st.pyplot(fig1)
        plt.close(fig1)

    # ===== TÍN HIỆU SAU LỌC =====

    with col2:

        st.subheader("2. Tín hiệu Sau lọc nhiễu")

        st.caption("Filtered Audio - Spectral Subtraction")

        st.audio(
            processed_audio,
            sample_rate=TARGET_SR
        )

        fig2, ax2 = plt.subplots(figsize=(7, 2.5))

        ax2.plot(
            time_axis,
            processed_audio,
            color="#16a085",
            linewidth=0.65
        )

        ax2.set_ylim([-1.05, 1.05])

        ax2.set_xlabel("Thời gian (s)", fontsize=9)
        ax2.set_ylabel("Biên độ", fontsize=9)

        ax2.set_title(
            "Filtered Audio Signal",
            fontsize=11,
            fontweight="bold"
        )

        ax2.grid(
            True,
            linestyle="--",
            alpha=0.25
        )

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        fig2.tight_layout()

        st.pyplot(fig2)
        plt.close(fig2)


    # =====================================================
    # 9. NHẬN DẠNG GIỌNG NÓI BẰNG WHISPER
    # =====================================================

    st.divider()

    st.header("🤖 Nhận dạng giọng nói bằng Whisper AI")

    st.caption(
        "Chuyển đổi tín hiệu âm thanh thành văn bản bằng mô hình Whisper."
    )

    if transcription_source == "Tín hiệu gốc":

        whisper_audio = raw_audio

    else:

        whisper_audio = processed_audio

    st.info(
        f"Whisper sẽ nhận diện từ: **{transcription_source}**"
    )

    if st.button(
        "🚀 Chuyển đổi Giọng nói thành Văn bản",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "🤖 Whisper đang phân tích âm thanh... "
                "Với bản ghi dài, quá trình có thể mất thêm thời gian."
            ):

                result = transcribe_speech(
                    whisper_audio,
                    model_size=model_size,
                    language=language
                )

                st.session_state.transcription_result = result

                st.session_state.transcription_source_used = (
                    transcription_source
                )

            st.success("Đã hoàn thành nhận dạng giọng nói!")

        except Exception as e:

            st.error(f"Lỗi khi nhận dạng giọng nói: {e}")


    # =====================================================
    # 10. HIỂN THỊ KẾT QUẢ NHẬN DẠNG
    # =====================================================

    result = st.session_state.transcription_result

    if result is not None:

        st.markdown("---")

        st.subheader("📝 Kết quả nhận dạng")

        st.caption(
            f"Nguồn nhận diện: {st.session_state.transcription_source_used}"
        )

        if result["text"]:

            st.success("✅ Nhận dạng giọng nói thành công!")

            st.text_area(
                "Văn bản nhận diện được:",
                value=result["text"],
                height=180,
                key="recognized_text"
            )

            # ---------------------------------------------
            # THÔNG TIN KẾT QUẢ
            # ---------------------------------------------

            segments = result.get("segments", [])

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Số phân đoạn nhận diện",
                len(segments)
            )

            result_col2.metric(
                "Số ký tự văn bản",
                len(result["text"])
            )

            # ---------------------------------------------
            # TIMESTAMPS
            # ---------------------------------------------

            with st.expander(
                "🔍 Xem chi tiết từng phân đoạn thời gian (Timestamps)"
            ):

                if segments:

                    for index, seg in enumerate(segments, start=1):

                        st.markdown(
                            f"**Đoạn {index} "
                            f"[{seg['start']:.2f}s → "
                            f"{seg['end']:.2f}s]:** "
                            f"{seg['text']}"
                        )

                else:

                    st.caption("Không có thông tin phân đoạn.")

            # ---------------------------------------------
            # TẢI VĂN BẢN
            # ---------------------------------------------

            st.download_button(
                label="📄 Tải văn bản nhận diện (.txt)",
                data=result["text"],
                file_name="ket_qua_nhan_dang.txt",
                mime="text/plain",
                use_container_width=True
            )

        else:

            st.warning(
                "Không nhận diện được giọng nói nào trong đoạn thu. "
                "Hãy kiểm tra âm thanh hoặc thử lại với tín hiệu gốc."
            )


# =========================================================
# 11. FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div style="
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    padding: 10px;
">
    <b>DSP Speech-to-Text System</b>
    <br>
    Digital Signal Processing | Speech Recognition | OpenAI Whisper
</div>
""", unsafe_allow_html=True)
