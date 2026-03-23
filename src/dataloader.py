import os
import random
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

# Mengimpor dari preprocessing.py (Pastikan file ini ada)
# from src.preprocessing import DynamicNoiseInjector 

class SEDDataset(Dataset):
    def __init__(self, df, config_data, is_train=True, augmentor=None, mlb=None):
        self.df = df
        self.config = config_data
        self.is_train = is_train
        self.augmentor = augmentor
        self.mlb = mlb # MultiLabelBinarizer dari scikit-learn
        
        self.sr = config_data.get('sample_rate', 32000)
        self.duration = config_data.get('max_duration', 5.0)
        self.audio_length = int(self.sr * self.duration) # Jumlah sampel mutlak (misal 160.000)
        
        # Base directory untuk audio yang sudah diproses (folder 'processed')
        self.processed_dir = config_data.get('processed_dir', '/kaggle/working/vigilant-octo-enigma/data/processed')

        # Transformasi Torchaudio ke Mel-Spectrogram
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=config_data.get('n_fft', 2048),
            hop_length=config_data.get('hop_length', 512),
            n_mels=config_data.get('n_mels', 128),
            f_min=config_data.get('fmin', 20),
            f_max=config_data.get('fmax', 16000),
            power=2.0
        )
        
        # Transformasi Amplitudo ke Decibel (Log Scale)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Audio (menggunakan kolom 'filepath' dari prepare_splits.py)
        audio_path = os.path.join(self.processed_dir, row['filepath'])
        waveform, current_sr = torchaudio.load(audio_path)
        
        # Konversi ke Mono jika Stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample jika diperlukan
        if current_sr != self.sr:
            resampler = torchaudio.transforms.Resample(current_sr, self.sr)
            waveform = resampler(waveform)

        # 2. RANDOM CROPPING ATAU PADDING (Defensive Programming)
        waveform_length = waveform.shape[1]
        
        if waveform_length > self.audio_length:
            if self.is_train:
                start = random.randint(0, waveform_length - self.audio_length)
            else:
                start = 0 
            waveform = waveform[:, start : start + self.audio_length]
        
        elif waveform_length < self.audio_length:
            padding = self.audio_length - waveform_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # 3. AUGMENTASI NOISE (Opsional)
        if self.augmentor is not None and self.is_train:
            waveform_np = waveform.squeeze(0).numpy()
            waveform_np = self.augmentor(waveform_np=waveform_np, sample_rate=self.sr)
            waveform = torch.tensor(waveform_np).unsqueeze(0)

        # 4. EKSTRAKSI FITUR (Waveform 1D -> Spectrogram 2D)
        mel_spec = self.mel_spectrogram(waveform)
        image_features = self.amplitude_to_db(mel_spec)

        # 5. PEMBUATAN TARGET MULTI-LABEL
        # Pecah label berdasarkan titik koma (;) untuk menangani soundscapes
        raw_labels = str(row['primary_label']).split(';')
        labels = [lbl.strip() for lbl in raw_labels if lbl.strip()]
                
        # Transformasi label menjadi tensor multi-hot (misal: [0., 1., 0., 0., ...])
        target = self.mlb.transform([labels])[0]
        target = torch.tensor(target, dtype=torch.float32)

        return {
            "input_values": image_features, # Gambar Spectrogram 2D (1 Channel)
            "labels": target               # Array Biner (234 kelas)
        }

def get_dataloader(config_data):
    print("\n[DataLoader] Inisialisasi Data pipeline...")
    
    # 1. Load Data yang sudah di-split secara offline
    train_csv_path = config_data.get('train_split_csv', '/kaggle/working/vigilant-octo-enigma/data/fixed/train_split.csv')
    val_csv_path   = config_data.get('val_split_csv', '/kaggle/working/vigilant-octo-enigma/data/fixed/val_split.csv')
    
    train_df = pd.read_csv(train_csv_path)
    val_df   = pd.read_csv(val_csv_path)
    
    print(f"[DataLoader] Memuat {len(train_df)} sampel Train dan {len(val_df)} sampel Validasi.")

    # 2. Setup MultiLabelBinarizer (234 Spesies)
    taxonomy_csv = config_data.get('taxonomy_csv', '/kaggle/input/birdclef-2026/taxonomy.csv')
    if not os.path.exists(taxonomy_csv):
        taxonomy_csv = '/kaggle/input/competitions/birdclef-2026/taxonomy.csv'
        
    taxonomy_df = pd.read_csv(taxonomy_csv)
    all_classes = sorted(taxonomy_df['primary_label'].unique())

    mlb = MultiLabelBinarizer(classes=all_classes)
    mlb.fit([all_classes])
    print(f"[DataLoader] MLB terpasang untuk {len(mlb.classes_)} kelas unik.")

    # 3. Setup Dataset PyTorch
    train_ds = SEDDataset(
        train_df, 
        config_data, 
        is_train=True, 
        augmentor=None, # Isi dengan noise_injector jika Anda ingin memakai augmentasi
        mlb=mlb
    )
    
    val_ds = SEDDataset(
        val_df, 
        config_data, 
        is_train=False, 
        augmentor=None, 
        mlb=mlb
    )

    # 4. Setup DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=config_data['batch_size'],
        shuffle=True, 
        num_workers=config_data.get('num_workers', 2),
        pin_memory=True,
        drop_last=True # Mencegah error jika sisa batch terakhir berukuran 1
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config_data['batch_size'],
        shuffle=False,
        num_workers=config_data.get('num_workers', 2),
        pin_memory=True
    )

    return train_loader, val_loader