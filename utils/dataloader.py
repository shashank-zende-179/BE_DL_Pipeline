import os
from PIL import Image
from torch.utils.data import Dataset

class LunarDataset(Dataset):
    def __init__(self, low_path, gt_path, transform=None):
        self.low_path = low_path
        self.gt_path = gt_path
        self.files = sorted(os.listdir(low_path))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        low_img = Image.open(os.path.join(self.low_path, img_name)).convert("RGB")
        gt_img = Image.open(os.path.join(self.gt_path, img_name)).convert("RGB")

        if self.transform:
            low_img = self.transform(low_img)
            gt_img = self.transform(gt_img)

        return low_img, gt_img