import numpy as np

# =====================================================================
# 1. HÀM CỬA SỔ HAMMING
# =====================================================================
def my_hamming(N: int) -> np.ndarray:
    """
    Tạo cửa sổ Hamming chiều dài N theo đúng công thức toán học:
    w[n] = 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
    """
    n = np.arange(N)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    return window.astype(np.float32)


# =====================================================================
# 2. THUẬT TOÁN FFT & IFFT (Cooley-Tukey Radix-2 - Tự code bằng NumPy)
# =====================================================================
def my_fft(x: np.ndarray) -> np.ndarray:
    """
    Thuật toán FFT Cooley-Tukey Radix-2 đệ quy (Vector hóa).
    Hỗ trợ cả mảng 1D (chiều dài N) hoặc ma trận 2D (num_frames, N).
    Yêu cầu: N phải là lũy thừa của 2 (ví dụ: 256, 512, 1024).
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[-1]

    # Kiểm tra N có phải lũy thừa của 2 không
    if N & (N - 1) != 0:
        raise ValueError(f"Chiều dài khung N={N} phải là lũy thừa của 2 (256, 512, 1024...)")

    # Điều kiện dừng đệ quy
    if N <= 1:
        return x

    # Chia thành phần tử vị trí Chẵn (Even) và Lẻ (Odd) ở trục cuối cùng
    X_even = my_fft(x[..., 0::2])
    X_odd = my_fft(x[..., 1::2])

    # Tính hệ số xoay pha (Twiddle factors): W_N^k = exp(-j * 2 * pi * k / N)
    k = np.arange(N // 2)
    twiddle = np.exp(-2j * np.pi * k / N)

    # Công thức cánh bướm (Butterfly operation) ghép 2 nửa
    first_half = X_even + twiddle * X_odd
    second_half = X_even - twiddle * X_odd

    return np.concatenate([first_half, second_half], axis=-1)


def my_ifft(X: np.ndarray) -> np.ndarray:
    """
    Biến đổi Fourier nhanh ngược (Inverse FFT) dựa trên tính chất đối ngẫu:
    IFFT(X) = (1 / N) * conj(FFT(conj(X)))
    """
    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[-1]
    return np.conj(my_fft(np.conj(X))) / N


# =====================================================================
# 3. BIẾN ĐỔI STFT (Miền Thời Gian -> Miền Tần Số - Thời Gian)
# =====================================================================
def my_stft(audio: np.ndarray, n_fft: int = 512, hop_length: int = 256) -> np.ndarray:
    """
    Chia khung (Framing), nhân cửa sổ Hamming (Windowing) và biến đổi FFT.
    Trả về ma trận phức STFT có kích thước: (num_frames, n_fft // 2 + 1)
    """
    audio = np.asarray(audio, dtype=np.float32)

    # Đệm thêm số 0 (Zero-padding) ở đầu và cuối để không bị mất mẫu ở rìa
    pad_len = n_fft // 2
    audio_padded = np.pad(audio, (pad_len, pad_len), mode='reflect')

    # Tính tổng số khung (frames)
    num_frames = 1 + (len(audio_padded) - n_fft) // hop_length

    # Chia khung (Framing) và nhân cửa sổ Hamming
    window = my_hamming(n_fft)
    frames = np.zeros((num_frames, n_fft), dtype=np.float32)

    for m in range(num_frames):
        start = m * hop_length
        frames[m, :] = audio_padded[start : start + n_fft] * window

    # Gọi hàm my_fft tự code cho toàn bộ các khung
    full_stft = my_fft(frames)

    # Vì tín hiệu âm thanh là số thực, phổ FFT đối xứng qua tần số Nyquist (n_fft // 2).
    # Chỉ cần giữ lại nửa phổ dương (từ 0 đến n_fft // 2) để tiết kiệm bộ nhớ.
    stft_matrix = full_stft[:, : n_fft // 2 + 1]
    return stft_matrix


# =====================================================================
# 4. BIẾN ĐỔI NGƯỢC ISTFT & OVERLAP-ADD (Miền Tần Số -> Miền Thời Gian)
# =====================================================================
def my_istft(stft_matrix: np.ndarray, orig_len: int, n_fft: int = 512, hop_length: int = 256) -> np.ndarray:
    """
    Khôi phục lại tín hiệu âm thanh 1D từ ma trận STFT bằng phương pháp Overlap-Add (Cộng chồng lấp).
    """
    num_frames = stft_matrix.shape[0]

    # Khôi phục lại nửa phổ âm (đối xứng liên hợp phức - Hermitian symmetry) trước khi chạy IFFT
    # Bỏ qua bin 0 (DC) và bin cuối (Nyquist) khi lấy đối xứng
    negative_freqs = np.conj(stft_matrix[:, -2:0:-1])
    full_stft = np.concatenate([stft_matrix, negative_freqs], axis=-1)

    # Biến đổi ngược IFFT về các khung miền thời gian (lấy phần thực)
    time_frames = np.real(my_ifft(full_stft))

    # Chuẩn bị mảng đầu ra và mảng chuẩn hóa năng lượng cửa sổ
    pad_len = n_fft // 2
    total_len = (num_frames - 1) * hop_length + n_fft
    reconstructed = np.zeros(total_len, dtype=np.float64)
    window_sum = np.zeros(total_len, dtype=np.float64)

    window = my_hamming(n_fft)

    # Cộng chồng lấp (Overlap-Add) có trọng số cửa sổ
    for m in range(num_frames):
        start = m * hop_length
        reconstructed[start : start + n_fft] += time_frames[m, :] * window
        window_sum[start : start + n_fft] += window ** 2

    # Chia chuẩn hóa cho tổng bình phương hàm cửa sổ (tránh chia cho 0)
    non_zero = window_sum > 1e-10
    reconstructed[non_zero] /= window_sum[non_zero]

    # Cắt bỏ phần đệm (padding) ban đầu để trả về đúng chiều dài gốc orig_len
    reconstructed = reconstructed[pad_len : pad_len + orig_len]
    return reconstructed.astype(np.float32)


# =====================================================================
# 5. THUẬT TOÁN LỌC NHIỄU TRỪ PHỔ (SPECTRAL SUBTRACTION - Tự code NumPy)
# =====================================================================
def spectral_subtraction_filter(
    noisy_audio: np.ndarray, 
    sr: int = 16000, 
    n_fft: int = 512, 
    hop_length: int = 256,
    noise_frames: int = 15,
    alpha: float = 2.0,
    beta: float = 0.02
) -> np.ndarray:
    """
    Lọc nhiễu âm thanh bằng thuật toán Trừ phổ công suất (Berouti's Spectral Subtraction).
    - noisy_audio: Mảng âm thanh 1D có nhiễu (chuẩn 16kHz).
    - noise_frames: Số khung đầu tiên dùng để ước lượng phổ nhiễu nền (15 khung ~ 0.25 giây).
    - alpha: Hệ số trừ quá mức (Over-subtraction factor, thường từ 1.5 đến 3.0).
    - beta: Hệ số sàn phổ (Spectral floor, thường từ 0.01 đến 0.05) chống nhiễu Musical Noise.
    """
    orig_len = len(noisy_audio)

    # Bước 1: Biến đổi tín hiệu sang miền Thời gian - Tần số bằng hàm my_stft tự code
    stft_noisy = my_stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)

    # Bước 2: Tách Phổ biên độ (Magnitude) và Phổ pha (Phase)
    magnitude = np.abs(stft_noisy)
    phase = np.angle(stft_noisy)  # Tương đương e^(j * phi) sau này dùng np.exp(1j * phase)

    # Tính phổ công suất (Power Spectrum = Magnitude^2)
    power_spec = magnitude ** 2

    # Bước 3: Ước lượng phổ công suất nhiễu trung bình từ các khung đầu tiên (Noise Profile)
    num_available_frames = min(noise_frames, power_spec.shape[0] // 4)
    num_available_frames = max(1, num_available_frames)
    noise_power_est = np.mean(power_spec[:num_available_frames, :], axis=0, keepdims=True)

    # Bước 4: Trừ phổ theo công thức Berouti (kết hợp Alpha và sàn phổ Beta)
    subtracted_power = power_spec - alpha * noise_power_est
    spectral_floor = beta * power_spec
    clean_power = np.maximum(subtracted_power, spectral_floor)

    # Lấy căn bậc hai để quay về Phổ biên độ sạch (Clean Magnitude)
    clean_magnitude = np.sqrt(clean_power)

    # Bước 5: Làm mượt nhẹ phổ theo trục thời gian (Moving average 3 khung) để giảm méo giọng
    clean_magnitude_smooth = clean_magnitude.copy()
    clean_magnitude_smooth[1:-1, :] = (
        0.25 * clean_magnitude[:-2, :] + 
        0.50 * clean_magnitude[1:-1, :] + 
        0.25 * clean_magnitude[2:, :]
    )

    # Bước 6: Tái tạo ma trận STFT phức (Ghép biên độ sạch + Pha gốc) và chạy my_istft
    stft_clean = clean_magnitude_smooth * np.exp(1j * phase)
    clean_audio = my_istft(stft_clean, orig_len=orig_len, n_fft=n_fft, hop_length=hop_length)

    return clean_audio


# =====================================================================
# 6. HÀM PHỤ TRỢ: TRỘN NHIỄU TRẮNG AWGN THEO MỨC SNR (0dB, 5dB, 10dB)
#    VÀ ĐO ĐỘ CẢI THIỆN SNR ĐỂ KIỂM CHỨNG BỘ LỌC
# =====================================================================
def add_awgn_noise(clean_audio: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Chèn nhiễu trắng Gaussian (AWGN) vào tín hiệu sạch theo đúng mức SNR (dB) chỉ định.
    """
    signal_power = np.mean(clean_audio ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    
    # Sinh nhiễu Gaussian có trung bình = 0, độ lệch chuẩn = sqrt(noise_power)
    noise = np.random.normal(0, np.sqrt(noise_power), size=clean_audio.shape).astype(np.float32)
    return clean_audio + noise


def calculate_snr(clean_ref: np.ndarray, test_signal: np.ndarray) -> float:
    """
    Tính tỷ số Tín hiệu trên Nhiễu (SNR) tính bằng dB giữa tín hiệu tham chiếu sạch và tín hiệu kiểm tra.
    """
    noise_residual = test_signal - clean_ref
    p_signal = np.sum(clean_ref ** 2)
    p_noise = np.sum(noise_residual ** 2)
    if p_noise < 1e-12:
        return 100.0
    return float(10 * np.log10(p_signal / p_noise))

# =====================================================================
# 7. THUẬT TOÁN TRÍCH XUẤT ĐẶC TRƯNG MFCC (Tự code 100% bằng NumPy)
# =====================================================================
def create_mel_filterbank(sr: int = 16000, n_fft: int = 512, n_mels: int = 26) -> np.ndarray:
    """
    Tạo ma trận Ngân hàng bộ lọc tam giác Mel (Mel Filterbank) kích thước (n_mels, n_fft // 2 + 1)
    """
    f_min = 0.0
    f_max = sr / 2.0  # Tần số Nyquist (8000 Hz)

    # 1. Đổi Hz sang Mel: m = 2595 * log10(1 + f / 700)
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)

    # 2. Chia đều(n_mels + 2) điểm trên trục Mel
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # 3. Đổi ngược từ Mel về Hz: f = 700 * (10^(m / 2595) - 1)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

    # 4. Ánh xạ tần số Hz sang chỉ số vạch phổ FFT (FFT bin index)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # 5. Dựng các bộ lọc hình tam giác
    num_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, num_bins), dtype=np.float32)

    for i in range(1, n_mels + 1):
        f_left = bin_points[i - 1]
        f_center = bin_points[i]
        f_right = bin_points[i + 1]

        # Sườn lên của tam giác
        for k in range(f_left, f_center):
            if f_center != f_left:
                filterbank[i - 1, k] = (k - f_left) / (f_center - f_left)
        # Sườn xuống của tam giác
        for k in range(f_center, f_right):
            if f_right != f_center:
                filterbank[i - 1, k] = (f_right - k) / (f_right - f_center)

    return filterbank


def compute_mfcc(
    audio: np.ndarray, 
    sr: int = 16000, 
    n_fft: int = 512, 
    hop_length: int = 256, 
    n_mels: int = 26, 
    n_mfcc: int = 13,
    pre_emphasis: float = 0.97
) -> np.ndarray:
    """
    Trích xuất ma trận đặc trưng MFCC bằng NumPy thuần.
    Trả về ma trận kích thước: (n_mfcc, num_frames) để dễ dàng vẽ biểu đồ nhiệt (Heatmap).
    """
    audio = np.asarray(audio, dtype=np.float32)

    # Bước 1: Lọc thông cao Pre-emphasis: y[n] = x[n] - 0.97 * x[n-1]
    emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # Bước 2: Biến đổi STFT (dùng hàm my_stft tự code ở Bước 1) và tính Phổ công suất
    stft_mat = my_stft(emphasized_audio, n_fft=n_fft, hop_length=hop_length)
    power_frames = (np.abs(stft_mat) ** 2) / n_fft  # Shape: (num_frames, n_fft // 2 + 1)

    # Bước 3: Nhân với Ngân hàng bộ lọc Mel (Mel Filterbank)
    mel_filters = create_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_energy = np.dot(power_frames, mel_filters.T)  # Shape: (num_frames, n_mels)

    # Bước 4: Lấy Logarit tự nhiên (chèn 1e-10 để tránh lỗi log(0))
    log_mel_energy = np.log(np.maximum(mel_energy, 1e-10))

    # Bước 5: Biến đổi Cosine rời rạc (DCT Type-II) tự code bằng nhân ma trận NumPy
    # Công thức: DCT[n, i] = sqrt(2 / M) * cos(pi * n * (2*i + 1) / (2 * M))
    n = np.arange(n_mfcc)[:, np.newaxis]       # Shape: (n_mfcc, 1)
    i = np.arange(n_mels)[np.newaxis, :]       # Shape: (1, n_mels)
    dct_matrix = np.sqrt(2.0 / n_mels) * np.cos(np.pi * n * (2 * i + 1) / (2.0 * n_mels))
    dct_matrix[0, :] *= 1.0 / np.sqrt(2.0)     # Chuẩn hóa trực giao cho hệ số c_0

    # Nhân ma trận Log-Mel với ma trận DCT
    mfcc_features = np.dot(log_mel_energy, dct_matrix.T)  # Shape: (num_frames, n_mfcc)

    # Chuyển vị thành (n_mfcc, num_frames) theo đúng chuẩn hiển thị phổ (trục tung là 13 hệ số MFCC, trục hoành là Thời gian)
    return mfcc_features.T.astype(np.float32)


# =====================================================================
# CHẠY KIỂM CHỨNG TOÀN BỘ BƯỚC 1 + BƯỚC 2 TRÊN TERMINAL
# =====================================================================
if __name__ == "__main__":
    import soundfile as sf
    
    print("=== KIỂM TRA BỘ LỌC NHIỄU SPECTRAL SUBTRACTION ===")
    sr = 16000
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
    
    # Tạo tín hiệu sạch giả lập: 0.4 giây đầu im lặng (để lấy mẫu nhiễu), sau đó là tổ hợp sóng âm
    clean_sig = np.zeros_like(t, dtype=np.float32)
    active_idx = int(0.4 * sr)
    clean_sig[active_idx:] = (
        0.5 * np.sin(2 * np.pi * 300 * t[active_idx:]) + 
        0.3 * np.sin(2 * np.pi * 800 * t[active_idx:])
    )
    
    # Thử chèn nhiễu trắng AWGN ở mức 5 dB (nhiễu khá nặng)
    noisy_sig = add_awgn_noise(clean_sig, snr_db=5.0)
    snr_before = calculate_snr(clean_sig, noisy_sig)
    
    # Đưa qua bộ lọc Spectral Subtraction tự code
    denoised_sig = spectral_subtraction_filter(noisy_sig, sr=sr, alpha=2.5, beta=0.01)
    snr_after = calculate_snr(clean_sig, denoised_sig)
    
    print(f"1. SNR trước khi lọc (Tín hiệu lẫn nhiễu AWGN): {snr_before:.2f} dB")
    print(f"2. SNR sau khi qua bộ lọc Spectral Subtraction: {snr_after:.2f} dB")
    print(f"👉 Độ cải thiện chất lượng tín hiệu (Delta SNR): +{snr_after - snr_before:.2f} dB!")
    
    # Nếu bạn đã có file 'test_mic_16khz.wav' từ hôm trước, lọc thử luôn file giọng nói thật!
    try:
        real_voice, _ = sf.read("test_mic_16khz.wav", dtype='float32')
        noisy_voice = add_awgn_noise(real_voice, snr_db=5.0)
        clean_voice = spectral_subtraction_filter(noisy_voice, sr=sr, alpha=2.0, beta=0.02)
        
        sf.write("1_giong_bi_nhieu_5dB.wav", noisy_voice, sr)
        sf.write("2_giong_da_loc_DSP.wav", clean_voice, sr)
        print("✅ Đã xuất 2 file '1_giong_bi_nhieu_5dB.wav' và '2_giong_da_loc_DSP.wav'!")
        print("   Hãy bấm vào nghe thử sự khác biệt trước và sau khi lọc!")
    except FileNotFoundError:
        print("(Gợi ý: Nếu có file test_mic_16khz.wav trong thư mục, code sẽ tự xuất file nghe thử!)")