
import torch 
import torch.nn as nn


class NoEncoder(nn.Module):
    def __init__(self):
        super(NoEncoder, self).__init__()
    
    @torch.no_grad()
    def forward(self, x):
        return x