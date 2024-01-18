import torch

from ..layers import get_tangent_plane_at_point, ln2_mlp_until_out, ln2_mlp_until_post

from transformer_lens import utils


def _validate_cache(cache, data, model, layer):
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
    cache = _validate_cache(cache, data, model, layer)
    split_vals = cache[utils.get_act_name("v", layer)]

    # 'batch src head d_head, head d_head d_model -> batch head src d_model'
    weighted_outs = torch.einsum("b s h f, h f m -> b h s m", split_vals, model.W_O[layer])

    # 'batch head src d_model, d_model -> batch head src'
    contribs = torch.einsum("b h s m, m -> b h s", weighted_outs, range_normal)

    return contribs

# def _get_feature_mid(
#     model, example, feature_token_idx, feature_post, use_ln=True, layer=0, mlp_out=True
# ):
#     with torch.no_grad():
#         _, cache = model.run_with_cache(
#             example, names_filter=[utils.get_act_name("resid_mid", layer)]
#         )
#     mid_acts = cache[utils.get_act_name("resid_mid", layer)]
#     x_mid = mid_acts[0, feature_token_idx][None, None, :]

#     my_fun = ln2_mlp_until_post if not mlp_out else ln2_mlp_until_out
#     feature_mid = get_tangent_plane_at_point(
#         x_mid, lambda x: my_fun(x, model.blocks[layer].ln2, model.blocks[layer].mlp, use_ln=use_ln), feature_post
#     )[0, 0]
#     return feature_mid


def attributions(model, feature_vector, example, token_idx, layer, use_ln=True, mlp_out=True):
    # Get cache
    _, cache = model.run_with_cache(
        example,
        names_filter=[
            # utils.get_act_name("post", layer),
            utils.get_act_name("resid_mid", layer),
            utils.get_act_name("attn_scores", layer),
        ],
    )
    # mlp_acts_flattened = cache[utils.get_act_name("post", lin.layer)].reshape(-1, SAE_CFG["d_mlp"])
    # _, _, hidden_acts, _, _ = lin.sae(mlp_acts_flattened)

    # Linearization component
    mid_acts = cache[utils.get_act_name("resid_mid", layer)]
    x_mid = mid_acts[0, token_idx][None, None, :]
    my_fun = ln2_mlp_until_out if mlp_out else ln2_mlp_until_post
    feature_mid = get_tangent_plane_at_point(
        x_mid, lambda x: my_fun(x, model.blocks[layer].ln2, model.blocks[layer].mlp, use_ln=False), feature_vector
    )[0, 0]
    attn_contribs = torch.cat(
        [_get_attn_head_contribs(model, l, range_normal=feature_mid, data=example) for l in range(layer + 1)],
        dim=0,
    )
    ov_contribs = torch.cat(
        [_get_attn_head_contribs_ov(model, l, range_normal=feature_mid, data=example) for l in range(layer + 1)],
        dim=0,
    )
    return {"attn": attn_contribs, "ov": ov_contribs}
