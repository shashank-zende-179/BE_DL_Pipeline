import torch.nn.functional as F

def hybrid_loss(output, target):
    l1 = F.l1_loss(output, target)
    mse = F.mse_loss(output, target)

    return l1 + 0.5 * mse

    import torch.nn.functional as F

# =========================
# Hybrid Loss Function
# =========================
def hybrid_loss(output, target):
    """
    Combines L1 and MSE loss for better performance
    """
    l1 = F.l1_loss(output, target)
    mse = F.mse_loss(output, target)

    return l1 + 0.5 * mse