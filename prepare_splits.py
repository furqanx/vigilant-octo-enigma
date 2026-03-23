import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def get_unique_labels(df):
    """
    Mengekstrak seluruh spesies unik dari DataFrame.
    Menangani kasus multi-label yang dipisah dengan titik koma (;).
    """
    unique_labels = set()
    for labels in df['primary_label'].dropna():
        # Pecah berdasarkan ';' lalu bersihkan spasi
        for label in str(labels).split(';'):
            unique_labels.add(label.strip())
    return unique_labels

def main():
    print("🚀 Memulai Proses Data Splitting & Safeguard...\n")
    
    # ==========================================
    # 1. SETUP DIREKTORI PENGKABELAN
    # ==========================================
    # Sesuaikan dengan path Kaggle working directory Anda
    BASE_DATA_DIR = "/kaggle/working/vigilant-octo-enigma/data"
    PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "processed")
    FIXED_DIR     = os.path.join(BASE_DATA_DIR, "fixed")
    os.makedirs(FIXED_DIR, exist_ok=True)
    
    # Path input
    focal_csv_path      = os.path.join(PROCESSED_DIR, "train_audio_cropped",       "cropped_train.csv")
    soundscape_csv_path = os.path.join(PROCESSED_DIR, "train_soundscapes_cropped", "cropped_soundscapes.csv")
    
    # ==========================================
    # 2. DATA MERGING & PATH STANDARDIZATION
    # ==========================================
    print("[1/4] Membaca dan Menggabungkan Data...")
    
    df_focal = pd.read_csv(focal_csv_path)
    # Tambahkan prefix folder agar Dataloader gampang mencarinya
    df_focal['filepath'] = "train_audio_cropped/" + df_focal['new_filename']
    df_focal['source']   = 'focal'
    
    df_soundscape = pd.read_csv(soundscape_csv_path)
    # Tambahkan prefix folder
    df_soundscape['filepath'] = "train_soundscapes_cropped/" + df_soundscape['new_filename']
    df_soundscape['source']   = 'soundscape'
    
    # ==========================================
    # 3. SPLITTING ANTI-BOCOR (BERBASIS GRUP)
    # ==========================================
    print("[2/4] Melakukan Group-Shuffle-Split (Anti-Leakage)...")
    
    # Split Data Focal (Train 95%, Val 5%)
    gss_focal = GroupShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    focal_train_idx, focal_val_idx = next(gss_focal.split(df_focal, groups=df_focal['filename']))
    
    train_focal = df_focal.iloc[focal_train_idx]
    val_focal   = df_focal.iloc[focal_val_idx]
    
    # Split Data Soundscape (Train 80%, Val 20%)
    gss_soundscape = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    sound_train_idx, sound_val_idx = next(gss_soundscape.split(df_soundscape, groups=df_soundscape['filename']))
    
    train_sound = df_soundscape.iloc[sound_train_idx]
    val_sound   = df_soundscape.iloc[sound_val_idx]
    
    # Gabungkan hasil split
    train_df = pd.concat([train_focal, train_sound], ignore_index=True)
    val_df   = pd.concat([val_focal, val_sound], ignore_index=True)
    
    print(f"      -> Train awal: {len(train_df)} potongan | Val awal: {len(val_df)} potongan")
    
    # ==========================================
    # 4. FAILSAFE SAFEGUARD (PENYELAMATAN SPESIES LANGKA)
    # ==========================================
    print("[3/4] Menjalankan Failsafe Safeguard (Memastikan tidak ada spesies buta)...")
    
    iteration = 1
    while True:
        train_labels = get_unique_labels(train_df)
        val_filenames_to_move = set()
        
        # Cek setiap file di Validation
        val_grouped = val_df.groupby('filename')
        for filename, group in val_grouped:
            group_labels = get_unique_labels(group)
            
            # Jika ada spesies di file ini yang TIDAK ADA di Train, tandai file ini!
            if not group_labels.issubset(train_labels):
                val_filenames_to_move.add(filename)
                
        # Jika tidak ada file bermasalah, keluar dari loop
        if not val_filenames_to_move:
            break
            
        print(f"      -> Iterasi {iteration}: Menyelamatkan {len(val_filenames_to_move)} file berisiko dari Validation ke Train!")
        
        # Pindahkan SEMUA potongan dari file-file tersebut ke Train
        moved_df = val_df[val_df['filename'].isin(val_filenames_to_move)]
        train_df = pd.concat([train_df, moved_df], ignore_index=True)
        
        # Hapus file-file tersebut dari Validation
        val_df = val_df[~val_df['filename'].isin(val_filenames_to_move)]
        
        iteration += 1

    # ==========================================
    # 5. EKSPOR HASIL AKHIR
    # ==========================================
    print("\n[4/4] Mengekspor Hasil Akhir ke folder data/fixed/...")
    
    train_out_path = os.path.join(FIXED_DIR, "train_split.csv")
    val_out_path   = os.path.join(FIXED_DIR, "val_split.csv")
    
    # Acak urutan baris di Train agar Dataloader membaca dengan urutan natural
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Hanya simpan kolom yang kita butuhkan agar CSV ringan
    columns_to_keep = ['filepath', 'primary_label', 'filename', 'source']
    
    train_df[columns_to_keep].to_csv(train_out_path, index=False)
    val_df[columns_to_keep].to_csv(val_out_path, index=False)
    
    print(f"✅ Selesai!")
    print(f"📊 Final Train Set : {len(train_df)} potongan (mengandung {len(get_unique_labels(train_df))} spesies unik)")
    print(f"📊 Final Val Set   : {len(val_df)} potongan (mengandung {len(get_unique_labels(val_df))} spesies unik)")
    print(f"📁 Lokasi Train : {train_out_path}")
    print(f"📁 Lokasi Val   : {val_out_path}")

if __name__ == "__main__":
    main()