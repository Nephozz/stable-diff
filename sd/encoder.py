import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            # (Batch_size, Channel, Heigh, Width) -> (Batch_size, 128, Heigh, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
        )
        