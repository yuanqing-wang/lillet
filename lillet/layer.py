import torch
import roma
import math

class Linear(torch.nn.Module):
    """ Linearly mixing the particles.

    Parameters
    ----------
    in_particles : int
        Number of input particles.

    out_particles : int
        Number of output particles.

    """
    def __init__(
            self,
            in_particles: int,
            out_particles: int,
            heads: int,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.randn(
                heads, in_particles, out_particles,
            )
        )

    def forward(
            self,
            X: torch.Tensor,
    ):
        return torch.einsum(
            "hio, hid -> hod",
            self.W.softmax(dim=-2),
            X,
        )
    
class Scale(torch.nn.Module):
    def __init__(
            self,
            heads: int,
            in_particles: int,
    ):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.randn(
                heads, in_particles,
            )
        )

    def forward(
            self,
            X: torch.Tensor,
    ):
        return X * self.W.tanh().unsqueeze(-1)
    


