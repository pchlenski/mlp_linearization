import torch

def get_tangent_plane_at_point(x_0_new, f, range_normal):
    """Linear approximation of f at x_0_new"""
    x_0_new.requires_grad_(True)
    g = lambda x: f(x) @ range_normal
    grad = torch.autograd.grad(g(x_0_new), x_0_new)
    return grad[0]


def ln2_mlp_until_post(x, ln, mlp, use_ln=True):
    """Get MLP activations for x"""
    if use_ln:
        x = ln(x)
    x = x @ mlp.W_in + mlp.b_in
    x = mlp.act_fn(x)
    return x


def ln2_mlp_until_out(x, ln, mlp, use_ln=True):
    """Get MLP outputs for x"""
    if use_ln:
        x = ln(x)
    return mlp(x)
