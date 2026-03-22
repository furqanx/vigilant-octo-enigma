import torch
import torch.nn as nn
import os
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

class SEDTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # --- MULTI-GPU SUPPORT ---
        if torch.cuda.device_count() > 1:
            print(f"🔥 Mengaktifkan {torch.cuda.device_count()} GPU dengan DataParallel!")
            self.model = torch.nn.DataParallel(self.model)
            
        self.output_dir = os.path.join(config['experiment']['output_dir'], config['experiment']['project_name'])
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Mixed Precision untuk VRAM efficiency
        self.scaler = torch.amp.GradScaler() if "cuda" in str(self.device) else None
        self.accum_steps = config['train'].get('gradient_accumulation_steps', 1)

        # --- LOSS FUNCTION MULTI-LABEL ---
        # BCEWithLogitsLoss menggabungkan Sigmoid dan Binary Cross Entropy dalam satu fungsi (lebih stabil)
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self):
        epochs = self.config['train']['epochs']
        best_auc = 0.0 
        
        print("\n[Trainer] Memulai Siklus Pelatihan Multi-Label SED...")
        for epoch in range(epochs):
            train_loss = self._train_epoch(epoch)
            
            print(f"\nEpoch {epoch+1} Selesai. Mengevaluasi Model...")
            val_loss, val_auc = self.validate()
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_auc:.4f}")
            
            self.save_checkpoint(epoch, val_auc, is_best=False)
            
            # Cek rekor baru (AUC lebih tinggi)
            if val_auc > best_auc:
                print(f"🔥 Rekor Baru! Menyimpan Model Terbaik (ROC-AUC: {best_auc:.4f} -> {val_auc:.4f})")
                best_auc = val_auc
                self.save_checkpoint(epoch, val_auc, is_best=True)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", leave=False)
        
        for step, batch in enumerate(pbar):
            input_values = batch['input_values'].to(self.device)
            labels = batch['labels'].to(self.device) # Shape: [Batch, 234]
            
            with torch.amp.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=(self.scaler is not None)):
                # 1. Forward Pass ke model (mendapatkan raw logits)
                logits = self.model(input_values)
                
                # 2. Hitung Multi-Label Loss
                loss = self.criterion(logits, labels)
                
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                    
                loss = loss / self.accum_steps

            # 3. Backward Pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 4. Optimizer & Scheduler Step
            if (step + 1) % self.accum_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step() # Di-update per batch!
                
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps
            pbar.set_postfix({'loss': loss.item() * self.accum_steps})

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        
        total_val_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch in pbar:
                input_values = batch['input_values'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Gunakan autocast juga di validasi agar cepat
                with torch.amp.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", enabled=(self.scaler is not None)):
                    logits = self.model(input_values)
                    loss = self.criterion(logits, labels)
                
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                    
                total_val_loss += loss.item()
                
                # Ubah logits menjadi probabilitas (0.0 sampai 1.0) menggunakan Sigmoid
                probs = torch.sigmoid(logits)
                
                all_preds.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        # Gabungkan semua batch menjadi satu array besar
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        avg_loss = total_val_loss / len(self.val_loader)

        # ====================================================
        # REVISI: PERHITUNGAN MACRO ROC-AUC YANG AMAN
        # ====================================================
        valid_aucs = []
        num_classes = all_targets.shape[1]

        for i in range(num_classes):
            # Hanya hitung AUC jika ada setidaknya satu label positif (1) dan satu negatif (0)
            if len(np.unique(all_targets[:, i])) > 1:
                class_auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                valid_aucs.append(class_auc)
                
        # Rata-ratakan skor AUC dari kelas-kelas yang valid
        if len(valid_aucs) > 0:
            auc_score = np.mean(valid_aucs)
        else:
            print("\n⚠️ Peringatan: Tidak ada kelas valid untuk menghitung AUC.")
            auc_score = 0.0

        return avg_loss, auc_score

    def save_checkpoint(self, epoch, metric, is_best=False):
        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'roc_auc': metric,
            'config': self.config
        }
        
        filename = "checkpoint_best.pth" if is_best else "checkpoint_last.pth"
        save_path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, save_path)