import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from audiomentations import AddBackgroundNoise

class AudioToMelSpectrogram:
    def __init__(self, config_data):
        """
        Inisialisasi ekstraktor fitur menjadi gambar 2D (Mel-spectrogram).
        Mengambil parameter matematis dari config.yaml.
        """
        self.sr = config_data.get('sample_rate', 32000)
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=config_data.get('n_fft', 2048),
            hop_length=config_data.get('hop_length', 512),
            n_mels=config_data.get('n_mels', 128),
            f_min=config_data.get('fmin', 50),
            f_max=config_data.get('fmax', 16000),
            power=2.0, # Power spectrogram
            normalized=True
        )
        # Mengubah skala daya ke Desibel (logaritmik) agar mirip persepsi pendengaran
        self.db_transform = T.AmplitudeToDB(top_db=80)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Input: Tensor 1D (Waveform)
        Output: Tensor 2D (Mel-spectrogram DB, ukuran: [1, n_mels, time_steps])
        """
        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)
        return mel_db

class DynamicNoiseInjector:
    def __init__(self, noise_dir, min_snr_db=3, max_snr_db=20, p=0.5):
        """
        Injeksi noise alam liar (suara latar Pantanal) ke rekaman bersih.
        """
        print(f"[Preprocessing] Menyiapkan Background Noise dari: {noise_dir}")
        self.augmentor = AddBackgroundNoise(
            sounds_path=noise_dir,
            min_snr_db=min_snr_db,
            max_snr_db=max_snr_db,
            p=p
        )

    def __call__(self, waveform_np: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.augmentor(samples=waveform_np, sample_rate=sample_rate)