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
        self.W_coarse_grain = torch.nn.Parameter(
            torch.randn(heads, coarse_grain_particles, fine_grain_particles)
        )

    def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.einsum("hfc, fd -> hc", self.W_fine_grain, x)
        x = torch.einsum("hfc, fd -> hc", self.W_fine_grain, h)
        return h, x
