import os
import random
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
import ast

# Mengimpor dari preprocessing.py
from src.preprocessing import DynamicNoiseInjector, AudioToMelSpectrogram

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
        self.audio_dir = config_data.get('train_audio_dir')

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
        file_col = 'new_filename' if 'new_filename' in row else 'filename'
        audio_path = os.path.join(self.audio_dir, row[file_col])
        
        # 1. Load Audio
        waveform, current_sr = torchaudio.load(audio_path)
        
        # Konversi ke Mono jika Stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample jika diperlukan
        if current_sr != self.sr:
            resampler = torchaudio.transforms.Resample(current_sr, self.sr)
            waveform = resampler(waveform)

        # 2. RANDOM CROPPING ATAU PADDING (Mencapai tepat 5 detik)
        waveform_length = waveform.shape[1]
        
        if waveform_length > self.audio_length:
            # Jika kepanjangan, potong acak 5 detik
            if self.is_train:
                start = random.randint(0, waveform_length - self.audio_length)
            else:
                start = 0 # Deterministic untuk validasi
            waveform = waveform[:, start : start + self.audio_length]
        
        elif waveform_length < self.audio_length:
            # Jika kependekan, tambahkan padding nol (suara hening)
            padding = self.audio_length - waveform_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # 3. AUGMENTASI NOISE (Pada level Waveform 1D)
        if self.augmentor is not None and self.is_train:
            waveform_np = waveform.squeeze(0).numpy()
            waveform_np = self.augmentor(waveform_np=waveform_np, sample_rate=self.sr)
            waveform = torch.tensor(waveform_np).unsqueeze(0)

        # 4. EKSTRAKSI FITUR (Ubah ke Gambar Mel-Spectrogram 2D)
        mel_spec = self.mel_spectrogram(waveform)
        image_features = self.amplitude_to_db(mel_spec)

        # if self.feature_extractor is not None:
        #     # Output shape: [1, n_mels, time_steps]
        #     image_features = self.feature_extractor(waveform)
        # else:
        #     image_features = waveform

        # 5. PEMBUATAN TARGET MULTI-LABEL
        # Menggabungkan primary_label dan secondary_label
        labels = [row['primary_label']]
        if 'secondary_labels' in row and pd.notna(row['secondary_labels']):
            # Convert string representasi list "[label1, label2]" menjadi list asli Python
            try:
                sec_labels = ast.literal_eval(row['secondary_labels'])
                labels.extend(sec_labels)
            except (ValueError, SyntaxError):
                pass
                
        # Gunakan MLB untuk mengubah list kategori menjadi tensor array probabilitas [0, 1, 0, 0, ...]
        target = self.mlb.transform([labels])[0]
        target = torch.tensor(target, dtype=torch.float32)

        return {
            "input_values": image_features, # Gambar 2D
            "labels": target               # Multi-hot vector
        }

def get_dataloader(config_data):
    print(f"[DataLoader] Menyiapkan dataset dari: {config_data['train_csv']}")
    df = pd.read_csv(config_data['train_csv'])
    
    # SETUP MULTILABEL BINARIZER (Daftar seluruh spesies kompetisi)
    # Sangat penting agar urutan indeks kelas (0-233) konsisten di seluruh tahap
    print(f"[DataLoader] Mengekstrak 234 kelas dari: {config_data['taxonomy_csv']}")
    taxonomy_df = pd.read_csv(config_data['taxonomy_csv'])

    all_classes = sorted(taxonomy_df['primary_label'].unique())

    mlb = MultiLabelBinarizer(classes=all_classes)
    mlb.fit([all_classes])
    
    # PEMBAGIAN DATASET (TRAIN, VAL, TEST) BEBAS KONTAMINASI
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_val_idx, test_idx = next(gss1.split(df, groups=df['filename']))
    
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # test_size 0.111 dari sisa 90% = ~10% dari total
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.111, random_state=42)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df['filename']))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    print(f"[DataLoader] Split Size -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # train_val_df, test_df = train_test_split(df, test_size=0.10, random_state=42)

    # train_df, val_df = train_test_split(train_val_df, test_size=0.111, random_state=42)

    # train_df = train_df.reset_index(drop=True)
    # val_df = val_df.reset_index(drop=True)
    # test_df = test_df.reset_index(drop=True)

    # print(f"[DataLoader] Split Size -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Inisialisasi Noise Injector (Opsional, pastikan folder noise_dir ada)
    noise_dir = config_data.get('noise_dir', None)
    noise_injector = None
    if noise_dir and os.path.exists(noise_dir):
        noise_injector = DynamicNoiseInjector(noise_dir=noise_dir, p=0.5)

    # Setup Dataset PyTorch
    train_ds = SEDDataset(
        train_df, 
        config_data, 
        is_train=True, 
        augmentor=noise_injector, 
        mlb=mlb
    )
    
    val_ds = SEDDataset(
        val_df, 
        config_data, 
        is_train=False, 
        augmentor=None, 
        mlb=mlb
    )

    test_ds = SEDDataset(
        test_df, 
        config_data, 
        is_train=False, 
        augmentor=None, 
        mlb=mlb
    )

    # Setup DataLoader (Cukup gunakan standar, tanpa Collator khusus)
    train_loader = DataLoader(
        train_ds,
        batch_size=config_data['batch_size'],
        shuffle=True, # Acak urutan file
        num_workers=config_data.get('num_workers', 2),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config_data['batch_size'],
        shuffle=False,
        num_workers=config_data.get('num_workers', 2),
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=config_data['batch_size'], 
        shuffle=False, 
        num_workers=config_data.get('num_workers', 2), 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader