import torch
import torch.nn as nn
import timm

class BirdCLEFModel(nn.Module):
    def __init__(self, model_name='tf_efficientnet_b0_ns', num_classes=234, pretrained=True, in_channels=1):
        """
        Arsitektur universal untuk BirdCLEF menggunakan timm.
        
        Args:
            model_name (str): Nama model dari library timm (contoh: 'tf_efficientnet_b0_ns', 'convnext_nano')
                - EfficientNet: 'tf_efficientnet_b0_ns' hingga 'tf_efficientnet_b4_ns' (Pro-tip: Akhiran '_ns' berarti Noisy Student weights, sangat superior untuk data audio yang berisik).
                - ConvNeXt: 'convnext_nano', 'convnext_tiny', 'convnext_small'
                - ResNeSt: 'resnest50d', 'resnest101e'
            num_classes (int): Jumlah spesies target (default: 234)
            pretrained (bool): Menggunakan bobot ImageNet (default: True)
            in_channels (int): Jumlah channel input (1 untuk Mel-Spectrogram grayscale)
        """
        super().__init__()
        self.model_name = model_name
        
        # ==========================================
        # 1. INISIALISASI BACKBONE (EKSTRAKTOR FITUR)
        # ==========================================
        # Parameter num_classes=0 adalah "cheat code" di timm untuk:
        # - Membuang layer klasifikasi (head) bawaan ImageNet.
        # - Tetap mempertahankan Global Average Pooling bawaan model tersebut.
        # Output dari backbone ini akan selalu berupa tensor 1D (vektor fitur).
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            in_chans=in_channels, 
            num_classes=0 
        )
        
        # ==========================================
        # 2. ADAPTASI DIMENSI DINAMIS
        # ==========================================
        # Mengambil otomatis ukuran dimensi output dari backbone yang dipilih
        # (Misal: 1280 untuk B0, 768 untuk ConvNeXt-Tiny, 2048 untuk ResNeSt50)
        in_features = self.backbone.num_features
        
        # ==========================================
        # 3. KEPALA KLASIFIKASI (CUSTOM HEAD)
        # ==========================================
        self.head = nn.Sequential(
            nn.Dropout(0.2),  # Regularisasi untuk mencegah model menghafal (overfitting)
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Alur maju (forward pass) jaringan.
        Input x shape: [Batch_Size, Channels(1), Freq_Bins(128), Time_Steps]
        """
        # Ekstraksi representasi fitur dari gambar spektrogram
        features = self.backbone(x)
        
        # Prediksi probabilitas logit untuk 234 kelas burung
        logits = self.head(features)
        
        return logits
    
def build_model(config_model):
    """
    Fungsi pembantu untuk menginisialisasi model berdasarkan parameter dari config.yaml.
    
    Args:
        config_model (dict): Dictionary yang berisi konfigurasi model.
                             Contoh: {'model_name': 'tf_efficientnet_b0_ns', 'num_classes': 234, ...}
    Returns:
        nn.Module: Model PyTorch yang siap digunakan.
    """
    # Mengambil nilai dari config dengan fallback ke nilai default yang aman
    model_name = config_model.get('model_name', 'tf_efficientnet_b0_ns')
    num_classes = config_model.get('num_classes', 234)
    pretrained = config_model.get('pretrained', True)
    in_channels = config_model.get('in_channels', 1)
    
    print(f"[Model] Membangun arsitektur : {model_name}")
    print(f"[Model] Pretrained Weights : {pretrained}")
    print(f"[Model] Input Channels     : {in_channels}")
    print(f"[Model] Output Classes     : {num_classes}")
    
    # Inisialisasi class utama
    model = BirdCLEFModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels
    )
    
    return model