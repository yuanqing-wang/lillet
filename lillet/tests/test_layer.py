import torch
HEADS = 4
IN_PARTICLES = 20
OUT_PARTICLES = 10


def test_linear_invariance(equivariance_test_utils):
    from lillet.layer import Linear
    translation, rotation, reflection = equivariance_test_utils
    layer = Linear(in_particles=IN_PARTICLES, out_particles=OUT_PARTICLES, heads=HEADS)
    x = torch.randn(HEADS, IN_PARTICLES, 3)
    y = layer(x)
    y_rotation = layer(rotation(x))
    y_reflection = layer(reflection(x))
    y_translation = layer(translation(x))
    assert torch.allclose(y_rotation, rotation(y), atol=1e-3, rtol=1e-3)
    assert torch.allclose(y_reflection, reflection(y), atol=1e-3, rtol=1e-3)
    assert torch.allclose(y_translation, translation(y), atol=1e-3, rtol=1e-3)

def test_spring_invariance(equivariance_test_utils):
    from lillet.layer import Spring
    translation, rotation, reflection = equivariance_test_utils
    layer = Spring()
    x = torch.randn(HEADS, IN_PARTICLES, 3)
    y = layer(x)
    y_rotation = layer(rotation(x))
    y_reflection = layer(reflection(x))
    y_translation = layer(translation(x))
    assert torch.allclose(y_rotation, rotation(y), atol=1e-3, rtol=1e-3)
    assert torch.allclose(y_reflection, reflection(y), atol=1e-3, rtol=1e-3)
    assert torch.allclose(y_translation, translation(y), atol=1e-3, rtol=1e-3)