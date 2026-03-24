import os
import yaml
import argparse
import torch
import torch.optim as optim

from src.dataloader import get_dataloader
from src.model import build_model
from src.trainer import SEDTrainer

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*50)
    print(f"🔄 MEMULAI PROSES RESUME TRAINING")
    print("="*50)

    # [REVISI 1]: Load Config dari file YAML terbaru (yang sudah di-sed), BUKAN dari checkpoint
    print(f"Membaca Konfigurasi terbaru dari: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Membaca Checkpoint dari: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    last_epoch  = checkpoint['epoch']
    start_epoch = last_epoch + 1
    best_auc    = checkpoint.get('roc_auc', 0.0) 

    print(f"✅ Checkpoint terbaca! Terakhir berhenti di Epoch {last_epoch + 1}.")
    print(f"✅ Rekor ROC-AUC tertinggi sebelumnya: {best_auc:.4f}")

    train_loader, val_loader = get_dataloader(config['data'])

    model = build_model(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print("✅ Otak Model (Weights) berhasil dipulihkan.")

    lr = float(config['train']['learning_rate'])

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=float(config['train'].get('weight_decay', 0.01)))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("✅ Momentum Optimizer berhasil dipulihkan.")

    accum_steps = int(config['train'].get('gradient_accumulation_steps', 1))
    max_train_steps = config['train']['epochs'] * (len(train_loader) // accum_steps)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=max_train_steps)

    if checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✅ Jadwal Learning Rate (Scheduler) berhasil dipulihkan.")

    trainer = SEDTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device, config=config
    )

    print("\n" + "="*50)
    print(f"🚀 MELANJUTKAN TRAINING DARI EPOCH {start_epoch + 1} HINGGA {config['train']['epochs']}")
    print("="*50)

    try:
        for epoch in range(start_epoch, config['train']['epochs']):
            train_loss = trainer._train_epoch(epoch)

            print(f"\nEpoch {epoch+1} Selesai. Mengevaluasi Model...")

            val_loss, val_auc = trainer.validate()

            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_auc:.4f}")

            trainer.save_checkpoint(epoch, val_auc, is_best=False)

            if val_auc > best_auc:
                print(f"🔥 Rekor Baru! Menyimpan Model Terbaik (ROC-AUC: {best_auc:.4f} -> {val_auc:.4f})")
                best_auc = val_auc
                trainer.save_checkpoint(epoch, val_auc, is_best=True)

    except KeyboardInterrupt:
        print("\n[Main] Training terinterupsi lagi. Tenang, progress terakhir aman di checkpoint_last.pth.")

    if start_epoch < config['train']['epochs']:
        print("\n[Main] Training Lanjutan Selesai. Menyimpan model final...")
        final_save_path = os.path.join(trainer.output_dir, "final_model")
        os.makedirs(final_save_path, exist_ok=True)
        model_save_file = os.path.join(final_save_path, "pytorch_model.pth")

        # [REVISI 2]: Gunakan trainer.model untuk memastikan sinkronisasi DataParallel aman
        final_state = trainer.model.module.state_dict() if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model.state_dict()
        torch.save(final_state, model_save_file)
        print(f"✅ Model final berhasil diamankan di: {model_save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume SED Training Script")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path menuju file checkpoint_last.pth')
    # [REVISI 3]: Tambahkan argumen config agar bisa membaca perubahan dari .yaml
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args)