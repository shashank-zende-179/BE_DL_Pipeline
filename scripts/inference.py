import torch
from PIL import Image
from torchvision import transforms

from models.hybrid_model import HybridEnhancer
from configs import config

# Load Model
model = HybridEnhancer().to(config.DEVICE)
model.load_state_dict(torch.load(config.HYBRID_SAVE_PATH))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load Image
img_path = "input.jpg"   # change this
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(config.DEVICE)

# Inference
with torch.no_grad():
    output = model(input_tensor)

# Save Output
output_img = output.squeeze().cpu()
transforms.ToPILImage()(output_img).save("output.jpg")

print("✅ Output saved as output.jpg")