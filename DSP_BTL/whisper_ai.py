from __future__ import annotations

import re
import whisper
import numpy as np
import torch


# ==========================================================
# 1. CACHE MÔ HÌNH WHISPER (TỰ ĐỘNG NHẬN DIỆN GPU / CPU)
# ==========================================================

_MODEL_CACHE = {}


def load_whisper_model(model_size: str = "small"):
    """
    Tải và lưu cache mô hình OpenAI Whisper.
    Tự động đẩy lên GPU (CUDA) nếu máy có card rời để tăng tốc.
    """
    if model_size not in _MODEL_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⏳ Đang tải mô hình Whisper '{model_size}' trên thiết bị [{device.upper()}]...")
        model = whisper.load_model(model_size, device=device)
        _MODEL_CACHE[model_size] = model
        print(f"✅ Đã tải xong mô hình '{model_size}'!")

    return _MODEL_CACHE[model_size]


# ==========================================================
# 2. TIỀN XỬ LÝ & TỐI ƯU TÍN HIỆU CHO WHISPER
# ==========================================================

def prepare_audio(audio_array: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Chuẩn hóa dữ liệu âm thanh đầu vào tối ưu cho bộ trích xuất Log-Mel của Whisper:
    - Chuyển về Mono float32
    - Khử DC Offset (thành phần một chiều)
    - Chuẩn hóa biên độ đỉnh (Peak Normalization về 0.95)
    - Đệm 0.25s khoảng lặng ở 2 đầu tránh mất chữ đầu/cuối câu
    """
    audio = np.asarray(audio_array)

    if audio.size == 0:
        raise ValueError("File âm thanh không có dữ liệu.")

    # 1. Gộp kênh về Mono nếu là Stereo
    if audio.ndim == 2:
        if audio.shape[0] <= 2:
            audio = np.mean(audio, axis=0)
        else:
            audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    # 2. Loại bỏ các giá trị NaN hoặc Inf (nếu có từ quá trình chia tần số DSP)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Khử DC Offset (đưa trung bình sóng âm về 0)
    audio = audio - np.mean(audio)

    # 4. Kiểm tra ngưỡng năng lượng tối thiểu
    max_amp = np.max(np.abs(audio))
    if max_amp < 1e-6:
        raise ValueError("Âm thanh quá nhỏ hoặc chỉ toàn khoảng lặng.")

    # 5. Chuẩn hóa biên độ đỉnh (Peak Normalization) về 0.95
    # Bước này cực kỳ quan trọng sau khi lọc trừ phổ DSP vì năng lượng tín hiệu thường bị giảm
    audio = (audio / max_amp) * 0.95

    # 6. Đệm 0.25 giây khoảng lặng ở đầu và cuối để Whisper không nuốt chữ đầu tiên/cuối cùng
    pad_samples = int(0.25 * sr)
    audio = np.pad(audio, (pad_samples, pad_samples), mode="constant", constant_values=0.0)

    return np.ascontiguousarray(audio, dtype=np.float32)


# ==========================================================
# 3. BỘ LỌC HẬU KỲ CHỐNG ẢO GIÁC (ANTI-HALLUCINATION)
# ==========================================================

_HALLUCINATION_PATTERNS = [
    r"hãy subscribe.*",
    r"đăng ký kênh.*",
    r"cảm ơn các bạn đã theo dõi.*",
    r"hẹn gặp lại các bạn.*",
    r"phụ đề được thực hiện bởi.*",
    r"amara\.org.*",
]


def _clean_transcribed_text(text: str) -> str:
    """Làm sạch khoảng trắng thừa và loại bỏ các câu ảo giác đặc trưng của Whisper."""
    cleaned = text.strip()
    for pattern in _HALLUCINATION_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    # Gộp nhiều khoảng trắng liên tiếp thành 1 khoảng trắng
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


# ==========================================================
# 4. NHẬN DIỆN GIỌNG NÓI CHÍNH XÁC CAO
# ==========================================================

def transcribe_speech(
    audio_array: np.ndarray,
    model_size: str = "small",
    language: str = "vi",
    custom_prompt: str | None = None
) -> dict:
    """
    Chuyển đổi âm thanh 16kHz Mono thành văn bản với cấu hình tối ưu độ chính xác.

    Trả về:
        {
            "text": Văn bản nhận diện đã làm sạch,
            "segments": Danh sách các phân đoạn kèm mốc thời gian
        }
    """
    model = load_whisper_model(model_size)
    audio_input = prepare_audio(audio_array)

    duration = len(audio_input) / 16000
    use_fp16 = torch.cuda.is_available()

    # Mồi ngữ cảnh (Initial Prompt) định hướng từ vựng và dấu câu chuẩn cho bộ giải mã
    if custom_prompt is not None:
        prompt_text = custom_prompt
    elif language == "vi":
        prompt_text = (
            "Xin chào, đây là bản ghi âm giọng nói tiếng Việt rõ ràng, chính xác, "
            "phục vụ bài tập lớn môn Xử lý tín hiệu số tại trường đại học."
        )
    else:
        prompt_text = "Hello, this is a clear and accurate English speech recording."

    print(f"🎙️ Đang nhận diện âm thanh ({duration:.2f}s) với model '{model_size}'...")

    # Cấu hình giải mã chuyên sâu
    result = model.transcribe(
        audio_input,
        language=language,
        task="transcribe",
        fp16=use_fp16,

        # Mồi ngữ cảnh giúp Whisper định hình chuẩn ngữ pháp & từ vựng tiếng Việt
        initial_prompt=prompt_text,

        # Mở rộng chùm tia tìm kiếm và độ kiên nhẫn khi giải mã
        beam_size=5,
        best_of=5,
        patience=1.5,

        # Fallback nhiệt độ: Ưu tiên 0.0 (chính xác nhất), tự động tăng nếu gặp đoạn âm thanh méo
        temperature=(0.0, 0.2, 0.4, 0.6),

        # TẮT phụ thuộc văn bản trước đó để tránh lỗi lặp chữ dây chuyền khi có nhiễu dư
        condition_on_previous_text=False,

        # Ngưỡng lọc khoảng lặng và kiểm soát chất lượng
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.2,
        logprob_threshold=-1.0,
        verbose=False
    )

    segments_result = result.get("segments", [])
    clean_segments = []
    valid_texts = []

    for seg in segments_result:
        # Bỏ qua các phân đoạn có xác suất là khoảng lặng/nhiễu quá cao
        if seg.get("no_speech_prob", 0.0) > 0.75:
            continue

        seg_text = _clean_transcribed_text(seg.get("text", ""))
        if not seg_text:
            continue

        # Trừ hao 0.25s đệm ở đầu để mốc thời gian khớp với âm thanh gốc
        start_time = max(0.0, float(seg.get("start", 0.0)) - 0.25)
        end_time = max(start_time, float(seg.get("end", 0.0)) - 0.25)

        clean_segments.append({
            "start": start_time,
            "end": end_time,
            "text": seg_text
        })
        valid_texts.append(seg_text)

    # Ghép từ các segment hợp lệ giúp loại bỏ rác ở đầu/cuối tốt hơn lấy result["text"] thô
    final_text = " ".join(valid_texts).strip()
    if not final_text:
        final_text = _clean_transcribed_text(result.get("text", ""))

    return {
        "text": final_text,
        "segments": clean_segments
    }


def transcribe_audio_array(
    audio_array: np.ndarray,
    model_size: str = "small",
    language: str = "vi"
) -> str:
    """
    Hàm bọc trả về chuỗi văn bản trực tiếp (Tương thích với code minh họa trong báo cáo).
    """
    res = transcribe_speech(audio_array, model_size=model_size, language=language)
    return res["text"]


# ==========================================================
# 5. CHẠY THỬ ĐỘC LẬP
# ==========================================================

if __name__ == "__main__":
    print("Đang kiểm tra hệ thống Whisper...")
    dummy_audio = np.zeros(16000 * 3, dtype=np.float32)

    try:
        result = transcribe_speech(dummy_audio, model_size="small", language="vi")
        print("Kết quả:", result["text"])
    except ValueError as e:
        print("Thông báo kiểm tra an toàn:", e)
    except Exception as e:
        print("Lỗi:", e)
