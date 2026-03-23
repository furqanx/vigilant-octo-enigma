import os
import yaml
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Impor dari modul lokal yang akan kita buat selanjutnya
from src.dataloader import get_dataloader
from src.model import build_model
from src.trainer import SEDTrainer 

import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

def set_seed(seed):
    """
    Mengatur seed agar hasil eksperimen bisa direproduksi (reproducible).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Setup] Seed set to: {seed}")

def main(args):

    # ====================================================
    # 1. SETUP & CONFIGURATION
    # ====================================================
    print(f"[Main] Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Buat folder output eksperimen
    output_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['project_name'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan config yang dipakai sebagai arsip
    with open(os.path.join(output_dir, "config_saved.yaml"), "w") as f:
        yaml.dump(config, f)

    set_seed(config['experiment']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")


    # ====================================================
    # 2. DATA PIPELINE
    # ====================================================
    # Logika Mel-spectrogram murni diurus di dalam Dataset (dataloader.py)
    train_loader, val_loader = get_dataloader(config['data'])
    print(f"[Main] Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


    # ====================================================
    # 3. BUILD MODEL (2D Vision Model)
    # ====================================================
    model = build_model(config['model'])
    model.to(device)


    # ====================================================
    # 4. OPTIMIZER & SCHEDULER (Dioptimalkan untuk Pretrained CNN)
    # ====================================================
    print("[Main] Configuring Optimizer & Scheduler...")
    
    lr           = float(config['train']['learning_rate'])
    weight_decay = float(config['train'].get('weight_decay', 0.01))
    epochs       = int(config['train']['epochs'])
    accum_steps  = int(config['train'].get('gradient_accumulation_steps', 1))

    optimizer    = optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )

    # Menghitung total langkah pembaruan (update steps)
    num_update_steps_per_epoch = len(train_loader) // accum_steps
    max_train_steps            = epochs * num_update_steps_per_epoch
    
    # REVISI: Menggunakan OneCycleLR (Mencakup Warmup + Cosine Decay)
    scheduler    = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max_train_steps,
        pct_start=0.1,           # 10% dari total langkah dialokasikan untuk fase Warmup
        anneal_strategy='cos',   # Kurva penurunan menggunakan Cosine (sama seperti CosineAnnealingLR)
        div_factor=25.0,         # LR dimulai dari (max_lr / 25) yang sangat kecil dan aman
        final_div_factor=10000.0 # LR berakhir di titik yang sangat mikroskopis
    )
    
    print(f"[Optimizer] Total Steps: {max_train_steps} | Scheduler: OneCycleLR (Warmup + Cosine)")


    # ====================================================
    # 5. INITIALIZE TRAINER (Fokus ke ROC-AUC)
    # ====================================================
    trainer = SEDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )


    # ====================================================
    # 6. START TRAINING Loop
    # ====================================================
    print("\n" + "="*50)
    print(f"🚀 STARTING TRAINING: {config['experiment']['project_name']}")
    print("="*50)
    
    try:
        trainer.train() 
    except KeyboardInterrupt:
        print("\n[Main] Training interrupted by user. Saving current state...")
    

    # ====================================================
    # 7. SAVE FINAL MODEL (PyTorch Native)
    # ====================================================
    print("\n[Main] Training Finished. Saving artifacts...")
    
    # Pastikan output_dir mengarah ke folder project yang benar
    project_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['project_name'])
    final_save_path = os.path.join(project_dir, "final_model")
    os.makedirs(final_save_path, exist_ok=True)
    
    # --- REVISI KRUSIAL: MULTI-GPU UNWRAPPING ---
    # Lepaskan bungkus DataParallel sebelum menyimpan agar format bobot tetap standar
    if isinstance(model, torch.nn.DataParallel):
        final_state_dict = model.module.state_dict()
    else:
        final_state_dict = model.state_dict()
    # --------------------------------------------
    
    # Menyimpan murni state_dict (bobot) dari model PyTorch
    model_save_file = os.path.join(final_save_path, "pytorch_model.pth")
    torch.save(final_state_dict, model_save_file)
    
    print(f"✅ Model weights berhasil diamankan di: {model_save_file}")
    print(f"✅ Saat inference, cukup inisialisasi arsitektur lalu gunakan model.load_state_dict()")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SED Training Script")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args)