import torch
from .radial import ExpNormalSmearing
from typing import Tuple
import torch

class Mapping(torch.nn.Module):
    def __init__(
            self,
            fine_grain_particles: int,
            coarse_grain_particles: int,
            heads: int = 1,
    ):
        super().__init__()
        self.fine_grain_particles = fine_grain_particles
        self.coarse_grain_particles = coarse_grain_particles
        self.heads = heads

class InductiveMapping(Mapping):
    def __init__(
            self,
            fine_grain_particles: int,
            coarse_grain_particles: int,
            heads: int = 1,
    ):
        super().__init__(
            fine_grain_particles=fine_grain_particles,
            coarse_grain_particles=coarse_grain_particles,
            heads=heads,
        )
        self.W_fine_grain = torch.nn.Parameter(
            torch.randn(heads, fine_grain_particles, coarse_grain_particles)
        )

    def forward(
            self,
            # h: torch.Tensor,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.einsum(
            "hfc, ...fd -> ...hcd", 
            self.W_fine_grain.softmax(-2), 
            x,
        )
        return x

class ReadDistance(torch.nn.Module):
    def __init__(
        self,
        smearing: torch.nn.Module = ExpNormalSmearing(),
    ):
        super().__init__()
        self.smearing = smearing
        
    def forward(self, x):
        # compute distances
        # (..., H, n, n, 3)
        delta_x = x.unsqueeze(-3) - x.unsqueeze(-2)

        # (..., H, n, n, 1)
        delta_x_norm = ((delta_x ** 2).sum(-1, keepdims=True) + 1e-5) ** 0.5

        # (..., H, n, n, N_RBF)
        delta_x_norm_smeared = self.smearing(delta_x_norm)

        return delta_x_norm_smeared





        
