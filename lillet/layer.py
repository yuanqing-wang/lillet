import torch
from .radial import ExpNormalSmearing
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
            "...hio, ...hit -> ...hot",
            self.W.softmax(dim=-2),
            X,
        )
        
class Spring(torch.nn.Module):
    def __init__(self, num_particles: int, heads: int):
        super().__init__()
        self.K = torch.nn.Parameter(torch.randn(heads, num_particles, num_particles))
        self.B = torch.nn.Parameter(torch.randn(heads, num_particles, num_particles))

    def forward(
            self,
            X: torch.Tensor,
    ):
        # compute distance
        delta_X = X.unsqueeze(-2) - X.unsqueeze(-3)
        distance = torch.norm(delta_X, dim=-1, keepdim=True)
        delta_X_direction = delta_X / (distance + EPSILON)

        # compute the force
        force_magnitude = self.K.unsqueeze(-1) * (distance - self.B.unsqueeze(-1))
        force = force_magnitude * delta_X_direction
        aggregated_force = torch.sum(force, dim=-2)
        X = X + aggregated_force
        return X
    
class Layer(torch.nn.Module):
    """ A layer of the model.

    Parameters
    ----------
    in_particles : int
        Number of input particles.

    out_particles : int
        Number of output particles.

    heads : int
        Number of heads.

    """
    def __init__(
            self,
            in_particles: int,
            out_particles: int,
            hidden_features: int,
            heads: int,
    ):
        super().__init__()
        self.linear = Linear(in_particles, out_particles, heads)
        self.spring = Spring(out_particles, heads)
        self.smearing = ExpNormalSmearing(num_rbf=hidden_features)

    def forward(
            self,
            X: torch.Tensor,
    ):
        X = self.linear(X)
        X = self.spring(X)
        H = self.smearing(X).flatten(-2, -1)
        return X, H

