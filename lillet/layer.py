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
        delta_X = X.unsqueeze(-2) - X.unsqueeze(-1)
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
            num_particles: int,
            num_dummies: int,
            num_heads: int, 
    ):
        super().__init__()
        self.W_left = torch.nn.Parameter(
            torch.randn(
                num_heads, num_particles, num_dummies,
            )
        )

        self.W_right = torch.nn.Parameter(
            torch.randn(
                num_heads, num_particles, num_dummies,
            )
        )

        self.fc = torch.nn.Linear(
            num_heads * num_particles * num_dummies,
            1,
        )

    def forward(
            self,
            X: torch.Tensor,
    ):
        # (NUM_HEADS, N, N, 3)
        delta_X = X.unsqueeze(-2) - X.unsqueeze(-1)

        # (NUM_HEADS, N, D, 3)
        X_left = torch.einsum(
            "habt, hbd -> hadt",
            delta_X,
            self.W_left,
        )

        X_right = torch.einsum(
            "habt, hbd -> hadt",
            delta_X,
            self.W_right,
        )

        # (NUM_HEADS, N, D)
        X_att = torch.einsum(
            "...at, ...at -> ...a",
            X_left,
            X_right,
        )

        X_att = X_att.flatten()



    

