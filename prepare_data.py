import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# ==========================================
# UTILITAS: PENCARI ENERGI TERTINGGI
# ==========================================
def get_best_5s_segment(y, sr, target_sec=5.0):
    """
    Mencari jendela 5 detik dengan total energi (RMS) tertinggi
    di dalam satu array audio.
    """
    target_samples = int(target_sec * sr)
    
    # Jika audio kurang dari atau sama dengan 5 detik, kembalikan apa adanya
    if len(y) <= target_samples:
        return y
        
    # Hitung Root Mean Square (RMS) energi
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    
    # Konversi 5 detik ke jumlah frame RMS
    frames_in_target = target_samples // hop_length
    
    # Hitung jumlah energi menggunakan moving window (convolution)
    window = np.ones(frames_in_target)
    rms_sums = np.convolve(rms, window, mode='valid')
    
    # Dapatkan indeks dengan energi tertinggi
    best_frame_idx = np.argmax(rms_sums)
    best_sample_idx = best_frame_idx * hop_length
    
    return y[best_sample_idx : best_sample_idx + target_samples]

def pad_if_needed(y, sr, target_sec=5.0):
    """Menambahkan keheningan (padding nol) jika audio kurang dari 5 detik."""
    target_samples = int(target_sec * sr)
    if len(y) < target_samples:
        pad_len = target_samples - len(y)
        y = np.pad(y, (0, pad_len), 'constant')
    return y

# ==========================================
# UTILITAS: KONVERSI WAKTU SUPER AMAN
# ==========================================
def time_str_to_seconds(time_val):
    """
    Mengonversi berbagai format waktu menjadi integer detik.
    Mendukung format: HH:MM:SS, MM:SS, float (5.0), atau integer (5).
    """
    # Jika sudah berupa angka murni
    if isinstance(time_val, (int, float)):
        return int(time_val)
        
    time_str = str(time_val).strip()
    
    # Jika string adalah angka (misal "5" atau "5.0")
    if time_str.replace('.', '', 1).isdigit():
        return int(float(time_str))
        
    # Jika menggunakan format titik dua
    parts = time_str.split(':')
    if len(parts) == 3: # HH:MM:SS
        h, m, s = map(float, parts)
        return int(h * 3600 + m * 60 + s)
    elif len(parts) == 2: # MM:SS
        m, s = map(float, parts)
        return int(m * 60 + s)
    else:
        raise ValueError(f"Format waktu tidak dikenali: {time_val}")

# ==========================================
# FUNGSI UTAMA 1: TRAIN AUDIO (Energy-Based)
# ==========================================
def process_train_audio(input_dir, output_dir, csv_path, sr=32000, block_sec=60.0, target_sec=5.0):
    print("\n[1] Memproses Train Audio (Energy-Based Cropping)...")
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    new_metadata = []
    
    block_samples = int(block_sec * sr)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Train Audio"):
        audio_path = os.path.join(input_dir, row['filename'])
        if not os.path.exists(audio_path):
            continue
            
        try:
            y, _ = librosa.load(audio_path, sr=sr)
        except Exception:
            continue
            
        primary_label = row['primary_label']
        base_name = os.path.splitext(os.path.basename(row['filename']))[0]
        
        # Buat folder untuk spesies ini di direktori output
        species_dir = os.path.join(output_dir, primary_label)
        os.makedirs(species_dir, exist_ok=True)
        
        # Kasus 1: Durasi < 5 detik
        if len(y) < int(target_sec * sr):
            y_padded = pad_if_needed(y, sr, target_sec)
            out_name = f"{base_name}_short.ogg"
            out_path = os.path.join(species_dir, out_name)
            sf.write(out_path, y_padded, sr)
            
            # REVISI: Menggunakan to_dict() untuk mencegah depresiasi Pandas
            new_row = row.to_dict()
            new_row['new_filename'] = f"{primary_label}/{out_name}"
            new_metadata.append(new_row)
            continue
            
        # Kasus 2 & 3: Durasi >= 5 detik (Potong per blok 60 detik)
        block_idx = 0
        for i in range(0, len(y), block_samples):
            block = y[i : i + block_samples]
            
            # Kasus 4: Abaikan sisa durasi di ujung file jika kurang dari 5 detik
            if len(block) < int(target_sec * sr):
                continue
                
            # Ekstrak 5 detik terbaik dari blok ini
            best_5s = get_best_5s_segment(block, sr, target_sec)
            
            out_name = f"{base_name}_block{block_idx}.ogg"
            out_path = os.path.join(species_dir, out_name)
            sf.write(out_path, best_5s, sr)
            
            # REVISI: Menggunakan to_dict()
            new_row = row.to_dict()
            new_row['new_filename'] = f"{primary_label}/{out_name}"
            new_metadata.append(new_row)
            
            block_idx += 1
            
    # Simpan metadata baru
    new_df = pd.DataFrame(new_metadata)
    new_df.to_csv(os.path.join(output_dir, "cropped_train.csv"), index=False)
    print(f"Selesai! Metadata baru disimpan di {output_dir}/cropped_train.csv")


# ==========================================
# FUNGSI UTAMA 2: TRAIN SOUNDSCAPES (Metadata-Driven)
# ==========================================
def process_train_soundscapes(input_dir, output_dir, csv_path, sr=32000):
    print("\n[2] Memproses Train Soundscapes (Metadata-Driven Cropping)...")
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    new_metadata = []
    
    grouped_df = df.groupby('filename')
    
    for filename, group in tqdm(grouped_df, desc="Processing Soundscapes"):
        audio_path = os.path.join(input_dir, filename)
        
        if not os.path.exists(audio_path):
            print(f"⚠️ File tidak ditemukan: {audio_path}")
            continue
            
        try:
            y, _ = librosa.load(audio_path, sr=sr)
        except Exception as e:
            print(f"⚠️ Gagal memuat {filename}: {e}")
            continue
            
        for _, row in group.iterrows():
            # 1. Ambil instruksi waktu menggunakan fungsi parsing yang sudah aman
            start_sec = time_str_to_seconds(row['start'])
            end_sec = time_str_to_seconds(row['end'])
            
            start_sample = start_sec * sr
            end_sample = end_sec * sr
            
            # Batasi indeks agar tidak melewati batas array audio
            start_sample = max(0, min(start_sample, len(y)))
            end_sample = max(0, min(end_sample, len(y)))
            
            chunk = y[start_sample:end_sample]
            
            # Jika potongan kosong, abaikan
            if len(chunk) == 0:
                continue
            
            base_name = os.path.splitext(filename)[0]
            out_name = f"{base_name}_{start_sec}_{end_sec}.ogg"
            out_path = os.path.join(output_dir, out_name)
            
            sf.write(out_path, chunk, sr)
            
            # REVISI: Menggunakan to_dict()
            new_row = row.to_dict()
            new_row['new_filename'] = out_name
            new_metadata.append(new_row)
            
    new_df = pd.DataFrame(new_metadata)
    new_csv_path = os.path.join(output_dir, "cropped_soundscapes.csv")
    new_df.to_csv(new_csv_path, index=False)
    print(f"Selesai! {len(new_df)} file diekstrak. Metadata disimpan di {new_csv_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    BASE_DIR        = "/kaggle/input/competitions/birdclef-2026"
    OUTPUT_BASE     = "/kaggle/working/vigilant-octo-enigma/data/processed"
    
    TRAIN_CSV_IN    = os.path.join(BASE_DIR,    "train.csv")
    TRAIN_AUDIO_IN  = os.path.join(BASE_DIR,    "train_audio")
    TRAIN_AUDIO_OUT = os.path.join(OUTPUT_BASE, "train_audio_cropped")
    
    # REVISI: Mendefinisikan lokasi CSV untuk soundscapes
    # Biasanya panitia menamainya train_soundscape_labels.csv atau soundscape_labels.csv
    SOUNDSCAPE_CSV  = os.path.join(BASE_DIR,    "train_soundscapes_labels.csv") 
    SOUNDSCAPE_IN   = os.path.join(BASE_DIR,    "train_soundscapes")
    SOUNDSCAPE_OUT  = os.path.join(OUTPUT_BASE, "train_soundscapes_cropped")
    
    # REVISI: Pastikan CSV soundscapes ada sebelum dijalankan dan dilempar ke fungsi
    if os.path.exists(SOUNDSCAPE_CSV):
        process_train_soundscapes(
            input_dir   = SOUNDSCAPE_IN, 
            output_dir  = SOUNDSCAPE_OUT,
            csv_path    = SOUNDSCAPE_CSV,  # <-- BUG FATAL DIPERBAIKI DISINI
            sr          = 32000
        )
    else:
        print(f"⚠️ PERINGATAN: File label soundscapes tidak ditemukan di {SOUNDSCAPE_CSV}. Proses Soundscapes dilewati.")

    process_train_audio(
        input_dir   = TRAIN_AUDIO_IN, 
        output_dir  = TRAIN_AUDIO_OUT, 
        csv_path    = TRAIN_CSV_IN,
        sr          = 32000, 
        block_sec   = 60.0, 
        target_sec  = 5.0
    )

if __name__ == "__main__":
    main()