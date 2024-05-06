from lillet.layer import LilletLayer
from lillet.mapping import InductiveMapping
import torch

def test_layer():
    layer = LilletLayer(
        mapping=InductiveMapping(fine_grain_particles=20, coarse_grain_particles=8),
    )
    x = torch.randn(10, 20, 3)
    layer(x)