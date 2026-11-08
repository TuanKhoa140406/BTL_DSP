import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf

TARGET_SR = 16000  # Tần số lấy mẫu chuẩn cho DSP tiếng nói và OpenAI Whisper

def normalize_audio_array(audio_data: np.ndarray) -> np.ndarray:
    """
    Hàm phụ trợ: Đưa mảng âm thanh về kiểu float32 và chuẩn hóa biên độ về [-1.0, 1.0]
    """
    # 1. Ép kiểu về float32 (chuẩn đầu vào của Whisper)
    audio_data = audio_data.astype(np.float32)
    
    # 2. Loại bỏ thành phần một chiều (DC offset - nếu mic bị rò điện áp nền)
    audio_data = audio_data - np.mean(audio_data)
    
    # 3. Chuẩn hóa biên độ đỉnh (Peak Normalization) về [-1.0, 1.0]
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val
        
    return audio_data

def load_audio_from_file(file_path: str) -> tuple[np.ndarray, int]:
    """
    Nhiệm vụ 1A: Đọc file âm thanh (.wav, .mp3,...) và chuẩn hóa
    """
    # librosa.load tự động gộp Stereo -> Mono (mono=True) 
    # và tự động lấy mẫu lại (Resample) về đúng 16000 Hz (sr=TARGET_SR)
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    
    # Chuẩn hóa biên độ và kiểu dữ liệu
    audio = normalize_audio_array(audio)
    
    return audio, sr

def record_audio_from_mic(duration_sec: int = 5) -> tuple[np.ndarray, int]:
    """
    Nhiệm vụ 1B: Thu âm trực tiếp từ Micro trong khoảng thời gian duration_sec (giây)
    """
    print(f"🎙️ Đang thu âm trong {duration_sec} giây... Hãy nói vào Micro!")
    
    # Thu âm trực tiếp ở tần số 16000 Hz, 1 kênh (Mono), kiểu float32
    recording = sd.rec(
        int(duration_sec * TARGET_SR), 
        samplerate=TARGET_SR, 
        channels=1, 
        dtype='float32'
    )
    sd.wait()  # Chờ cho đến khi thu âm xong
    print("✅ Đã thu âm xong!")
    
    # Chuyển mảng 2D (N, 1) của sounddevice thành mảng 1D (N,)
    audio = recording.flatten()
    
    # Chuẩn hóa biên độ
    audio = normalize_audio_array(audio)
    
    return audio, TARGET_SR

# ==========================================================
# ĐOẠN CODE CHẠY THỬ (TEST) ĐỂ BẠN KIỂM TRA TRÊN MÁY
# ==========================================================
if __name__ == "__main__":
    # Thử thu âm 3 giây từ mic của bạn
    audio_mic, sr = record_audio_from_mic(duration_sec=3)
    
    # In ra các thông số kỹ thuật để kiểm tra xem đã đạt chuẩn chưa
    print("--- THÔNG SỐ TÍN HIỆU SAU CHUẨN HÓA ---")
    print(f"1. Tần số lấy mẫu (Sample Rate): {sr} Hz")
    print(f"2. Kích thước mảng (Shape): {audio_mic.shape} (16000 mẫu/giây x 3 giây = 48000 mẫu)")
    print(f"3. Kiểu dữ liệu (Data type): {audio_mic.dtype}")
    print(f"4. Biên độ [Min, Max]: [{np.min(audio_mic):.2f}, {np.max(audio_mic):.2f}]")
    
    # Lưu thử ra file .wav để bật lên nghe lại xem mic thu có rõ không
    sf.write("test_mic_16khz.wav", audio_mic, sr)
    print("Đã lưu thành công file 'test_mic_16khz.wav'!")