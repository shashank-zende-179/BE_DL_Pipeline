# =========================
# Fix Python Path
# =========================
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =========================
# Imports
# =========================
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms   # 🔥 NEW

from models.hybrid_model import HybridEnhancer
from utils.dataloader import LunarDataset
from utils.metrics import psnr, ssim
from configs import config


def main():
    print("🧪 Testing Started...\n")

    # =========================
    # Device
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Using Device:", device)

    # =========================
    # Output Folder
    # =========================
    output_dir = "outputs2"
    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # 🔥 TEST TRANSFORM (256×256)
    # =========================
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),   # 🔥 FORCE 256
        transforms.ToTensor()
    ])

    # =========================
    # Dataset
    # =========================
    test_dataset = LunarDataset(config.VAL_LOW, config.VAL_GT, test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"📊 Total Test Images: {len(test_dataset)}\n")

    # =========================
    # Model
    # =========================
    model = HybridEnhancer(None, None).to(device)
    model.load_state_dict(torch.load(config.HYBRID_SAVE_PATH, map_location=device))
    model.eval()

    total_psnr = 0
    total_ssim = 0

    # =========================
    # Testing Loop
    # =========================
    with torch.no_grad():
        for i, (low, gt) in enumerate(test_loader):

            low = low.to(device)
            gt = gt.to(device)

            # Forward pass
            output = model(low)

            # Metrics
            img_psnr = psnr(output, gt)
            img_ssim = ssim(output, gt)

            total_psnr += img_psnr
            total_ssim += img_ssim

            print(f"Image {i+1}/{len(test_dataset)} → PSNR: {img_psnr:.2f}, SSIM: {img_ssim:.4f}")

            # =========================
            # Save Images (256×256)
            # =========================
            save_image(low, f"{output_dir}/input_{i+1}.png")
            save_image(output, f"{output_dir}/output_{i+1}.png")
            save_image(gt, f"{output_dir}/gt_{i+1}.png")

    # =========================
    # Final Results
    # =========================
    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)

    print("\n🎯 FINAL RESULTS")
    print(f"Average PSNR : {avg_psnr:.2f}")
    print(f"Average SSIM : {avg_ssim:.4f}")

    print(f"\n📁 Images saved in: {output_dir}/")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    main()