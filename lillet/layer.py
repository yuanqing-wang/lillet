import torch

class LilletLayer(torch.nn.Module):
    def __init__(
            self,
            mapping: torch.nn.Module,
    ):
        super().__init__()
        self.mapping = mapping

    def forward(
            self,
            h: torch.Tensor,
            x: torch.Tensor,
    ):
        h, x = self.mapping(h, x)
        return h, x