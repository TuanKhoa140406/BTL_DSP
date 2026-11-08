import os
import tempfile
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Import 3 module trong dự án của nhóm
from audio_io import load_audio_from_file, record_audio_from_mic, TARGET_SR
from dsp_core import spectral_subtraction_filter
from whisper_ai import load_whisper_model, transcribe_speech

# 1. Cấu hình trang Web
st.set_page_config(page_title="DSP Speech-to-Text", page_icon="🎙️", layout="wide")
st.title("🎙️ Hệ thống Nhận dạng Giọng nói kết hợp Lọc nhiễu DSP")
st.markdown("Bài tập lớn Xử lý Tín hiệu Số — Chuẩn hóa 16kHz, Lọc nhiễu & OpenAI Whisper")

# 2. Cache mô hình trên Streamlit để web chạy mượt không bị load lại
@st.cache_resource
def get_cached_model(size: str):
    return load_whisper_model(size)

# 3. Thanh cài đặt bên trái (Sidebar)
with st.sidebar:
    st.header("⚙️ Cấu hình Hệ thống")
    model_size = st.selectbox(
        "Chọn mô hình Whisper:",
        options=["tiny", "base", "small"],
        index=1,
        help="'tiny' nhanh nhất, 'small' chính xác hơn nhưng nặng hơn."
    )
    language = st.selectbox(
        "Ngôn ngữ nói:",
        options=[("Tiếng Việt", "vi"), ("Tiếng Anh", "en")],
        format_func=lambda x: x[0]
    )[1]
    
    apply_dsp = st.checkbox("Bật bộ lọc nhiễu DSP (Trừ phổ)", value=True)
    
    st.divider()
    st.caption("Trạng thái mô hình AI:")
    with st.spinner(f"Đang nạp Whisper ({model_size})..."):
        get_cached_model(model_size)
    st.success(f"Mô hình '{model_size}' đã sẵn sàng!")

# 4. Khởi tạo bộ nhớ phiên (Session State) để giữ âm thanh khi bấm các nút
if "raw_audio" not in st.session_state:
    st.session_state.raw_audio = None

# 5. Khu vực chọn nguồn âm thanh đầu vào (Chia 2 Tab)
tab_upload, tab_mic = st.tabs(["📂 Tải file âm thanh (.wav, .mp3)", "🎤 Thu âm trực tiếp từ Micro"])

with tab_upload:
    uploaded_file = st.file_uploader("Chọn file âm thanh từ máy tính", type=["wav", "mp3", "m4a", "flac"])
    if uploaded_file is not None:
        if st.button("📥 Đọc và Chuẩn hóa File", key="btn_load_file"):
            # Lưu ra file tạm để librosa đọc được đường dẫn
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            with st.spinner("Đang chuẩn hóa về 16kHz Mono..."):
                audio_data, _ = load_audio_from_file(tmp_path)
                st.session_state.raw_audio = audio_data
            os.remove(tmp_path)
            st.success("Đã tải và chuẩn hóa âm thanh thành công!")

with tab_mic:
    duration = st.slider("Thời lượng thu âm (giây):", min_value=2, max_value=15, value=5)
    if st.button("🔴 Bắt đầu Thu âm", type="primary"):
        with st.spinner(f"🎙️ Đang thu âm trong {duration} giây... Hãy nói vào Micro!"):
            audio_data, _ = record_audio_from_mic(duration_sec=duration)
            st.session_state.raw_audio = audio_data
        st.success("✅ Đã thu âm xong!")

# 6. Xử lý DSP, Vẽ đồ thị và Chạy Whisper khi đã có âm thanh
if st.session_state.raw_audio is not None:
    st.divider()
    raw_audio = st.session_state.raw_audio
    
    # Chạy qua bộ lọc DSP của bạn cùng nhóm
    if apply_dsp:
        processed_audio = spectral_subtraction_filter(raw_audio, sr=TARGET_SR)
    else:
        processed_audio = raw_audio.copy()

    # Chia 2 cột so sánh Trước và Sau khi lọc DSP
    col1, col2 = st.columns(2)
    time_axis = np.linspace(0, len(raw_audio) / TARGET_SR, num=len(raw_audio))

    with col1:
        st.subheader("1. Tín hiệu Gốc (Sau chuẩn hóa 16kHz)")
        st.audio(raw_audio, sample_rate=TARGET_SR)
        fig1, ax1 = plt.subplots(figsize=(6, 2))
        ax1.plot(time_axis, raw_audio, color="#1f77b4", linewidth=0.5)
        ax1.set_ylim([-1.05, 1.05])
        ax1.set_xlabel("Thời gian (s)")
        ax1.set_ylabel("Biên độ")
        st.pyplot(fig1)

    with col2:
        st.subheader("2. Tín hiệu Sau lọc nhiễu DSP")
        st.audio(processed_audio, sample_rate=TARGET_SR)
        fig2, ax2 = plt.subplots(figsize=(6, 2))
        ax2.plot(time_axis, processed_audio, color="#2ca02c", linewidth=0.5)
        ax2.set_ylim([-1.05, 1.05])
        ax2.set_xlabel("Thời gian (s)")
        ax2.set_ylabel("Biên độ")
        st.pyplot(fig2)

    # Nút bấm chạy nhận diện giọng nói Whisper
    st.divider()
    if st.button("🚀 Chuyển đổi Giọng nói thành Văn bản (Whisper AI)", type="primary", use_container_width=True):
        with st.spinner("🤖 Mô hình Whisper đang dịch âm thanh ra chữ..."):
            result = transcribe_speech(processed_audio, model_size=model_size, language=language)
        
        st.subheader("📝 Kết quả Văn bản:")
        if result["text"]:
            st.text_area("Văn bản nhận diện được:", value=result["text"], height=120)
            
            # Hiển thị chi tiết từng mốc thời gian (rất ăn điểm khi thuyết trình)
            with st.expander("🔍 Xem chi tiết từng phân đoạn thời gian (Timestamps)"):
                for seg in result["segments"]:
                    st.write(f"**[{seg['start']:.2f}s ➔ {seg['end']:.2f}s]:** {seg['text']}")
        else:
            st.warning("Không nhận diện được giọng nói nào trong đoạn thu. Hãy thử nói to và rõ hơn!")