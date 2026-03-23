import torch
import torch.nn as nn

# Import your models
from .LLFormer import LLFormer
from .restormer.RestormerArch import Restormer


class HybridEnhancer(nn.Module):
    def __init__(self, llformer_weights=None, restormer_weights=None):
        super(HybridEnhancer, self).__init__()

        # Initialize models
        self.llformer = LLFormer()
        self.restormer = Restormer()

        # -------------------------------
        # Load LLFormer weights
        # -------------------------------
        if llformer_weights is not None:
            ll_weights = torch.load(llformer_weights, map_location='cpu')

            # Handle BasicSR format
            if isinstance(ll_weights, dict) and 'params' in ll_weights:
                ll_weights = ll_weights['params']

            self.llformer.load_state_dict(ll_weights, strict=False)
            print("✅ LLFormer weights loaded")

        # -------------------------------
        # Load Restormer weights
        # -------------------------------
        if restormer_weights is not None:
            rs_weights = torch.load(restormer_weights, map_location='cpu')

            # Handle BasicSR format
            if isinstance(rs_weights, dict) and 'params' in rs_weights:
                rs_weights = rs_weights['params']

            self.restormer.load_state_dict(rs_weights, strict=False)
            print("✅ Restormer weights loaded")

    def forward(self, x):
        # -------------------------------
        # Step 1: Low-Light Enhancement
        # -------------------------------
        enhanced = self.llformer(x)

        # -------------------------------
        # Step 2: Denoising
        # -------------------------------
        output = self.restormer(enhanced)

        return output