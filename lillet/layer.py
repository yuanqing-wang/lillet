import torch
import roma
import math
EPSILON = 1e-5

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
    

class Spring(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.K = torch.nn.Parameter(torch.tensor(1.0))
        self.B = torch.nn.Parameter(torch.tensor(0.0))
        
    def forward(
            self,
            X: torch.Tensor,
    ):
        # compute distance
        delta_X = X[:, :, None] - X[:, None, :]
        distance = torch.norm(delta_X, dim=-1, keepdim=True)
        delta_X_direction = delta_X / (distance + EPSILON)

        # compute the force
        force_magnitude = self.K * (distance - self.B)
        force = force_magnitude * delta_X_direction
        aggregated_force = torch.sum(force, dim=-2)
        X = X + aggregated_force
        return X

class Outer(torch.nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
    

