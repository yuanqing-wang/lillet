from lillet.layer import LilletLayer
from lillet.mapping import InductiveMapping
import torch

def test_layer():
    layer = LilletLayer(
        mapping=InductiveMapping(fine_grain_particles=20, coarse_grain_particles=8),
    )
    x = torch.randn(10, 20, 3)
    layer(x)

def test_layer_invariance(equivariance_test_utils):
    translation, rotation, reflection = equivariance_test_utils
    layer = LilletLayer(
        mapping=InductiveMapping(fine_grain_particles=20, coarse_grain_particles=8),
    )
    x = torch.randn(20, 3)
    y = layer(x)
    y_translation = layer(translation(x))
    y_rotation = layer(rotation(x))
    y_reflection = layer(reflection(x))
    assert torch.allclose(y_translation, y)
    assert torch.allclose(y_rotation, y)
    assert torch.allclose(y_reflection, y)