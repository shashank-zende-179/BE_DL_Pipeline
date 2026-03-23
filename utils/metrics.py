import torch
import math

# =========================
# PSNR
# =========================
def psnr(output, target):
    mse = torch.mean((output - target) ** 2)

    if mse == 0:
        return 100

    return 20 * math.log10(1.0 / torch.sqrt(mse))


# =========================
# SSIM (Simplified Version)
# =========================
def ssim(output, target):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = output.mean()
    mu_y = target.mean()

    sigma_x = ((output - mu_x) ** 2).mean()
    sigma_y = ((target - mu_y) ** 2).mean()
    sigma_xy = ((output - mu_x) * (target - mu_y)).mean()

    return ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))