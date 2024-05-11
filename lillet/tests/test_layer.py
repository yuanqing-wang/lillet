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
    assert torch.allclose(y_rotation, rotation(y), atol=1e-3, rtol=1e-3)
    assert torch.allclose(y_reflection, reflection(y), atol=1e-3, rtol=1e-3)

def test_scale_invariance(equivariance_test_utils):
    from lillet.layer import Scale
    translation, rotation, reflection = equivariance_test_utils
    layer = Scale(in_particles=IN_PARTICLES, heads=HEADS)
    x = torch.randn(HEADS, IN_PARTICLES, 3)
    y = layer(x)
    y_rotation = layer(rotation(x))
    y_reflection = layer(reflection(x))
    assert torch.allclose(y_rotation, rotation(y), atol=1e-3, rtol=1e-3)
    assert torch.allclose(y_reflection, reflection(y), atol=1e-3, rtol=1e-3)

def test_linear_and_scale_invariance(equivariance_test_utils):
    from lillet.layer import Linear, Scale
    translation, rotation, reflection = equivariance_test_utils
    layer = Linear(in_particles=IN_PARTICLES, out_particles=OUT_PARTICLES, heads=HEADS)
    layer2 = Scale(in_particles=OUT_PARTICLES, heads=HEADS)
    x = torch.randn(HEADS, IN_PARTICLES, 3)
    y = layer2(layer(x))
    y_rotation = layer2(layer(rotation(x)))
    y_reflection = layer2(layer(reflection(x)))
    assert torch.allclose(y_rotation, rotation(y), atol=1e-3, rtol=1e-3)
    assert torch.allclose(y_reflection, reflection(y), atol=1e-3, rtol=1e-3)