import whisper
import numpy as np

# Biến toàn cục giúp lưu mô hình trên RAM, tránh việc mỗi lần bấm nút phải tải lại
_MODEL_CACHE = {}

def load_whisper_model(model_size: str = "base"):
    """
    Tải mô hình OpenAI Whisper vào bộ nhớ.
    Các kích thước: 'tiny', 'base', 'small', 'medium', 'large'
    - 'base' hoặc 'small' là cân bằng nhất khi chạy trên CPU laptop.
    """
    if model_size not in _MODEL_CACHE:
        print(f"⏳ Đang tải mô hình Whisper '{model_size}'...")
        _MODEL_CACHE[model_size] = whisper.load_model(model_size)
        print(f"✅ Đã tải xong mô hình '{model_size}'!")
    return _MODEL_CACHE[model_size]

def transcribe_speech(
    audio_array: np.ndarray, 
    model_size: str = "base", 
    language: str = "vi"
) -> dict:
    """
    Nhiệm vụ 2: Chuyển mảng âm thanh 16kHz float32 thành văn bản.
    Trả về dictionary chứa toàn bộ văn bản ('text') và từng phân đoạn ('segments').
    """
    model = load_whisper_model(model_size)
    
    # Đảm bảo mảng đầu vào đúng chuẩn float32
    audio_input = audio_array.astype(np.float32)
    
    # fp16=False giúp tránh cảnh báo/lỗi khi chạy trên CPU thông thường
    result = model.transcribe(
        audio_input, 
        language=language, 
        fp16=False
    )
    
    return {
        "text": result.get("text", "").strip(),
        "segments": result.get("segments", [])
    }

# ==========================================================
# ĐOẠN CODE CHẠY THỬ ĐỘC LẬP
# ==========================================================
if __name__ == "__main__":
    # Tạo 3 giây âm thanh im lặng giả lập để test xem Whisper chạy được chưa
    dummy_audio = np.zeros(16000 * 3, dtype=np.float32)
    res = transcribe_speech(dummy_audio, model_size="base", language="vi")
    print("Kết quả nhận diện thử:", res["text"])