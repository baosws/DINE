from typing import Any, List, Tuple
from torch import Tensor, nn
import torch
from torch.nn import functional as F

# https://github.com/e-hulten/maf/blob/2c0604ac8573ab14a6bc83dd51827d47a4266a96/made.py#L13
class MaskedLinear(nn.Linear):
    """Linear transformation with masked out elements. y = x.dot(mask*W.T) + b"""

    def __init__(self, n_in: int, n_out: int, mask: Tensor, bias: bool = True) -> None:
        """
        Args:
            n_in: Size of each input sample.
            n_out:Size of each output sample.
            bias: Whether to include additive bias. Default: True.
        """
        super().__init__(n_in, n_out, bias)
        self.register_buffer('mask', mask)

    def forward(self, x: Tensor) -> Tensor:
        """Apply masked linear transformation."""
        return F.linear(x, self.mask.T * self.weight, self.bias)

def get_mask(d_in, d_out, n_groups, mask_type, is_output):
    g_in, g_out = d_in // n_groups, d_out // n_groups
    x, y = torch.meshgrid(torch.arange(d_in), torch.arange(d_out), indexing='ij')
    x = torch.div(x, g_in, rounding_mode='floor')
    y = torch.div(y, g_out, rounding_mode='floor')
    if mask_type == 'autoregressive':
        if is_output:
            mask = (x < y).float()
        else:
            mask = (x <= y).float()
    elif mask_type == 'grouped':
        mask = (x == y).float()
    return mask

def MaskedMLP(d_in, d_out, n_groups, hidden_sizes, mask_type):
    if isinstance(hidden_sizes, int):
        hidden_sizes = [hidden_sizes]
    hidden_sizes = [d_in] + hidden_sizes + [d_out]
    act_func = lambda: nn.ReLU(inplace=True)
    layers = []
    for i in range(len(hidden_sizes) - 1):
        mask = get_mask(hidden_sizes[i], hidden_sizes[i + 1], n_groups, mask_type, is_output=(i == len(hidden_sizes) - 2))
        layers.append(MaskedLinear(hidden_sizes[i], hidden_sizes[i + 1], mask=mask))
        if i < len(hidden_sizes) - 2:
            layers.append(act_func())
        # layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))

    return nn.Sequential(*layers)