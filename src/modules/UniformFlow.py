import sys
sys.path.append('')
from torch import nn
import torch
from src.modules import MaskedMLP, MLP
from torch.distributions.normal import Normal

class UniformFlow(nn.Module):
    def __init__(self, d, dz, dh, n_components):
        super().__init__()
        self.Z_encoder = MLP(d_in=dz, d_out=dh, hidden_sizes=dh)
        self.conditioner = MaskedMLP(d_in=d, d_out=d * dh, n_groups=d, hidden_sizes=d, mask_type='autoregressive')
        self.c_to_gmm = MaskedMLP(d_in=d * dh, d_out=d * n_components * 3, n_groups=d, hidden_sizes=d * dh, mask_type='grouped')
        
        self.d = d
        self.dz = dz
        self.dh = dh
        self.n_components = n_components

    def forward(self, X, Z):
        N = X.shape[0]

        c = self.conditioner(X) + self.Z_encoder(Z).repeat(1, self.d) # N x [d x h]
        gmm = self.c_to_gmm(c).view(N, self.d, -1)
        mu = gmm[..., :self.n_components]
        std = gmm[..., self.n_components: 2 * self.n_components].exp()
        std = torch.clip(std, min=1e-6, max=None)
        logits = gmm[..., -self.n_components:]
        w = torch.softmax(logits, dim=2) # N x d x k

        dist = Normal(mu, std) # N x d x k
        e = (dist.cdf(X[..., None]) * w).sum(dim=2) # N x d
        p_hat = (dist.log_prob(X[..., None]).exp() * w).sum(dim=2)
        p_hat = torch.clip(p_hat, min=1e-24, max=None)
        # print(p_hat.min(), (1 / p_hat).max())
        log_de_dx = p_hat.log().sum(axis=1) # N
        assert log_de_dx.isnan().any() == False

        return e, log_de_dx