import os
import torch

# =========================
# 📂 PATH CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_LOW = os.path.join(DATA_DIR, "train/low_light_noisy")
TRAIN_GT = os.path.join(DATA_DIR, "train/ground_truth")

VAL_LOW = os.path.join(DATA_DIR, "val/low_light_noisy")
VAL_GT = os.path.join(DATA_DIR, "val/ground_truth")

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

LLFORMER_WEIGHTS = os.path.join(WEIGHTS_DIR, "model_bestSSIM.pth")
RESTORMER_WEIGHTS = os.path.join(WEIGHTS_DIR, "restormer_weights.pth")

HYBRID_SAVE_PATH = os.path.join(WEIGHTS_DIR, "hybrid_best.pth")


# =========================
# ⚙️ TRAINING PARAMETERS
# =========================
BATCH_SIZE = 4
NUM_WORKERS = 2
EPOCHS = 20
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 🖼️ IMAGE SETTINGS
# =========================
IMAGE_SIZE = 256
CHANNELS = 3


# =========================
# 🔁 MODEL SETTINGS
# =========================
LOAD_PRETRAINED = True   # Load LLFormer + Restormer weights
SAVE_BEST_ONLY = True


# =========================
# 📊 LOGGING
# =========================
PRINT_FREQ = 10   # print every N batches


# =========================
# 📈 METRICS
# =========================
USE_PSNR = True
USE_SSIM = True


# =========================
# 🧪 EXPERIMENT NAME
# =========================
EXPERIMENT_NAME = "LLFormer_Restormer_Hybrid"