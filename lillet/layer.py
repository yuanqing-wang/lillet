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
            * self.mapping.heads 
            * (self.mapping.coarse_grain_particles ** 4)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
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
        delta_x_norm = (delta_x ** 2).sum(-1, keepdims=True).relu() ** 0.5

        # (..., H, n, n, N_RBF)
        delta_x_norm_smeared = self.smearing(delta_x_norm)
        delta_x_norm_smeared = delta_x_norm_smeared / (delta_x_norm + 1e-6) ** 2

        # (..., H, n, n, N_RBF, 3)
        delta_x_basis = delta_x.unsqueeze(-2) * delta_x_norm_smeared.unsqueeze(-1)

        # (..., H, n ** 2, N_RBF, 3)
        delta_x_basis = delta_x_basis.reshape(
            *delta_x_basis.shape[:-4], 
            -1, 
            *delta_x_basis.shape[-2:],
        )

        # (..., H, n ** 2, n ** 2, N_RBF)
        att = torch.einsum(
            "...anx, ...bnx -> ...abn",
            delta_x_basis,
            delta_x_basis,
        )

        # (..., H, n ** 4, N_RBF)
        att = att.reshape(*att.shape[:-4], -1)
        return self.fc(att)








        
