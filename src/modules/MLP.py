from typing import Any, List, Tuple
from torch import nn
import torch

def MLP(d_in: int, d_out: int, hidden_sizes: List[int]=None):
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    hidden_sizes = [d_in] + (hidden_sizes or []) + [d_out]

    layers = []
    for i in range(len(hidden_sizes) - 1):
        layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        if i < len(hidden_sizes) - 2:
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
    
    return nn.Sequential(*layers)