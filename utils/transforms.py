from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),   # 🔥 reduced from 256
    transforms.ToTensor()
])


# =========================
# Optional (for future use)
# =========================
def get_transform(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])