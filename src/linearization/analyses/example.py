import torch
import transformer_lens
from typing import Dict
from ..sae_tutorial import AutoEncoder
from ..layers import get_tangent_plane_at_point, ln2_mlp_until_out, ln2_mlp_until_post

from transformer_lens import utils


def example_scores(
    model: transformer_lens.HookedTransformer,
    sae: AutoEncoder,
    act_name: str,
    layer: int,
    example: torch.Tensor,
    feature_idx: int,
    **absorb,
) -> Dict[str, torch.Tensor]:
    """
    Given an example tensor, return SAE feature scores for each token in that example.

    Args:
        model: HookedTransformer model
        example: Example tensor, shape (seq_len,)

    Returns:
        A Tensor of SAE feature scores, shape (seq_len,)
    """

    # Get cache
    _, cache = model.run_with_cache(example, names_filter=[utils.get_act_name(act_name, layer)])
    mlp_acts = cache[utils.get_act_name(act_name, layer)]
    hidden = sae(mlp_acts)[2]

    return hidden[:, :, feature_idx]


def get_feature_mid(model, example, token_idx, feature_vector, layer, use_ln, mlp_out, **absorb):
    # Get cache
    with torch.no_grad():
        _, cache = model.run_with_cache(example, names_filter=[utils.get_act_name("resid_mid", layer)])

    # Linearization component
    mid_acts = cache[utils.get_act_name("resid_mid", layer)]
    x_mid = mid_acts[0, token_idx][None, None, :]
    my_fun = ln2_mlp_until_out if mlp_out else ln2_mlp_until_post
    feature_mid = get_tangent_plane_at_point(
        x_mid, lambda x: my_fun(x, model.blocks[layer].ln2, model.blocks[layer].mlp, use_ln=use_ln), feature_vector
    )[0, 0]

    return feature_mid


def _validate_cache(cache, data, model, layer):
    """
    Helper function: validate cache and data
    """
    # Get cache
    if cache is None:
        if data is not None:
            _, cache = model.run_with_cache(
                data,
                # stop_at_layer=layer + 1,
                names_filter=[utils.get_act_name("pattern", layer), utils.get_act_name("v", layer)],
            )
        else:
            raise ValueError("Either cache or data must be provided.")
    return cache


def _get_attn_head_contribs(model, layer, range_normal, cache=None, data=None):
    """
    Helper function: get attention head contributions for a layer, prompt
    """
    cache = _validate_cache(cache, data, model, layer)
    split_vals = cache[utils.get_act_name("v", layer)]
    attn_pattern = cache[utils.get_act_name("pattern", layer)]

    # Ensure dtypes match
    if split_vals.dtype != attn_pattern.dtype:
        attn_pattern = attn_pattern.to(split_vals.dtype)

    #'batch head dst src, batch src head d_head -> batch head dst src d_head'
    weighted_vals = torch.einsum("b h d s, b s h f -> b h d s f", attn_pattern, split_vals)

    # 'batch head dst src d_head, head d_head d_model -> batch head dst src d_model'
    weighted_outs = torch.einsum("b h d s f, h f m -> b h d s m", weighted_vals, model.W_O[layer])

    # 'batch head dst src d_model, d_model -> batch head dst src'
    contribs = torch.einsum("b h d s m, m -> b h d s", weighted_outs, range_normal)

    return contribs


def _get_attn_head_contribs_ov(model, layer, range_normal, cache=None, data=None):
    """
    Helper fucntion: get OV-circuit attention head contributions for a layer, prompt
    """
    cache = _validate_cache(cache, data, model, layer)
    split_vals = cache[utils.get_act_name("v", layer)]

    # 'batch src head d_head, head d_head d_model -> batch head src d_model'
    weighted_outs = torch.einsum("b s h f, h f m -> b h s m", split_vals, model.W_O[layer])

    # 'batch head src d_model, d_model -> batch head src'
    contribs = torch.einsum("b h s m, m -> b h s", weighted_outs, range_normal)

    return contribs


def attributions(
    model: transformer_lens.HookedTransformer,
    feature_mid: torch.Tensor,
    # feature_vector: torch.Tensor,
    example: torch.Tensor,
    # token_idx: int,
    layer: int,
    # use_ln: bool = False,
    # mlp_out: bool = True,
    **absorb,
) -> Dict[str, torch.Tensor]:
    """
    Given an example tensor, return attention head and OV-circuit attributions for that prompt.

    Args:
        model: HookedTransformer model
        feature_vector: Feature vector for linearization, shape (d_model,)
        example: Example tensor, shape (seq_len,)
        token_idx: Index of token to linearize
        layer: Layer to linearize
        use_ln: Whether to use layer normalization
        mlp_out: Whether to use MLP outputs or activations

    Returns:
        A dict of attention head and OV-circuit attributions:
        {
            "attn": shape (n_heads, seq_len, seq_len)
            "ov": shape (n_heads, seq_len)
        }
    """
    attn_contribs = torch.cat(
        [_get_attn_head_contribs(model, l, range_normal=feature_mid, data=example) for l in range(layer + 1)],
        dim=0,
    )
    ov_contribs = torch.cat(
        [_get_attn_head_contribs_ov(model, l, range_normal=feature_mid, data=example) for l in range(layer + 1)],
        dim=0,
    )
    return {"attn": attn_contribs.detach().cpu().squeeze(), "ov": ov_contribs.detach().cpu().squeeze()}
