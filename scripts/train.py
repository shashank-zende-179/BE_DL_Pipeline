# =========================
# Fix Python Path
# =========================
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =========================
# CUDA Memory Optimization
# =========================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# =========================
# Multiprocessing Fix
# =========================
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

# =========================
# Imports
# =========================
import torch
from torch.utils.data import DataLoader

from models.hybrid_model import HybridEnhancer
from utils.dataloader import LunarDataset
from utils.transforms import transform
from utils.loss import hybrid_loss
from utils.metrics import psnr, ssim
from configs import config


def main():
    print("🔥 Training Started...")

    # =========================
    # DEVICE (GPU AUTO)
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using Device:", device)

    if device.type == "cuda":
        print("🔥 GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # =========================
    # Dataset
    # =========================
    train_dataset = LunarDataset(config.TRAIN_LOW, config.TRAIN_GT, transform)
    val_dataset = LunarDataset(config.VAL_LOW, config.VAL_GT, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,        # 🔥 IMPORTANT (was causing OOM)
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"📊 Train Samples: {len(train_dataset)}")
    print(f"📊 Val Samples: {len(val_dataset)}")

    # =========================
    # Model
    # =========================
    model = HybridEnhancer(None, None).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 🔥 Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    best_ssim = 0

    # =========================
    # Training Loop
    # =========================
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for i, (low, gt) in enumerate(train_loader):
            low = low.to(device)
            gt = gt.to(device)

            # 🔥 AMP Forward
            with torch.cuda.amp.autocast():
                output = model(low)
                loss = hybrid_loss(output, gt)

            # Backprop
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            # 🔥 Clear memory
            torch.cuda.empty_cache()

            if i % config.PRINT_FREQ == 0:
                print(f"[Epoch {epoch+1}/{config.EPOCHS}] Step {i} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # =========================
        # Validation
        # =========================
        model.eval()
        total_psnr = 0
        total_ssim = 0

        with torch.no_grad():
            for low, gt in val_loader:
                low = low.to(device)
                gt = gt.to(device)

                output = model(low)

                total_psnr += psnr(output, gt)
                total_ssim += ssim(output, gt)

        avg_psnr = total_psnr / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)

        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"Loss: {avg_loss:.4f}")
        print(f"PSNR: {avg_psnr:.2f}")
        print(f"SSIM: {avg_ssim:.4f}\n")

        # =========================
        # Save Best Model
        # =========================
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim

            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), config.HYBRID_SAVE_PATH)

            print("✅ Best Hybrid Model Saved!\n")

    print("🎯 Training Completed!")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    main()