import torch
from .radial import ExpNormalSmearing
from .mapping import InductiveMapping

class LilletLayer(torch.nn.Module):
    def __init__(
            self,
            mapping: torch.nn.Module,
            smearing: torch.nn.Module = ExpNormalSmearing(),
            activation: torch.nn.Module = torch.nn.SiLU(),
            hidden_features: int = 128,
    ):
        super().__init__()
        self.mapping = mapping
        self.smearing = smearing
        in_features = (
            self.smearing.num_rbf
            * (self.mapping.coarse_grain_particles ** 2)
        )

        self.fc_basis_left = torch.nn.Linear(in_features, hidden_features, bias=False)
        self.fc_basis_right = torch.nn.Linear(in_features, hidden_features, bias=False)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, 1),
        )

    def forward(
            self,
            x: torch.Tensor,
    ):
        # (..., H, n, 3)
        x = self.mapping(x)

        # compute distances
        # (..., H, n, n, 3)
        delta_x = x.unsqueeze(-3) - x.unsqueeze(-2)

        # (..., H, n, n, 1)
        delta_x_norm = ((delta_x ** 2).sum(-1, keepdims=True) + 1e-5) ** 0.5

        # (..., H, n, n, N_RBF)
        delta_x_norm_smeared = self.smearing(delta_x_norm)
        delta_x_norm_smeared = delta_x_norm_smeared / (delta_x_norm + 1e-5) ** 2

        # (..., H, n, n, N_RBF, 3)
        delta_x_basis = delta_x.unsqueeze(-2) * delta_x_norm_smeared.unsqueeze(-1)

        # (..., H, n ** 2 * N_RBF, 3)
        delta_x_basis = delta_x_basis.reshape(
            *delta_x_basis.shape[:-4], 
            -1, 
            delta_x_basis.shape[-1],
        )

        # (..., H, D, 3)
        delta_x_basis_left = self.fc_basis_left(delta_x_basis.swapaxes(-2, -1)).swapaxes(-2, -1)
        delta_x_basis_right = self.fc_basis_right(delta_x_basis.swapaxes(-2, -1)).swapaxes(-2, -1)

        # (..., H, D)
        # att = delta_x_basis.pow(2).sum(-1) + 1e-5
        # att = (delta_x_basis_left * delta_x_basis_right).sum(-1) + 1e-5
        att = torch.einsum(
            '...ab, ...ab -> ...a',
            delta_x_basis_left,
            delta_x_basis_right,
        ) + 1e-5

        att = att.logsumexp(-2)
        return self.fc(att)








        
