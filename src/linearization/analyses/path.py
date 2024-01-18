import transformer_lens
import torch

from transformer_lens import utils
from Typing import List, Tuple, Dict

from ..layers import ln2_mlp_until_out, ln2_mlp_until_post, get_tangent_plane_at_point


def feature_vectors(
    model: transformer_lens.HookedTransformer,
    example: torch.Tensor,
    token_idx: int,
    start_vector: torch.Tensor,
    path: List[Tuple],
    mlp_out: bool,
    **absorb
) -> Dict[str, List[torch.Tensor]]:
    """
    Given a path and an example, get feature vectors and de-embeddings after pulling back through each path component.

    Args:
        model: HookedTransformer model
        example: Example tensor, shape (seq_len,)
        token_idx: Index of token to linearize
        start_vector: Feature vector for linearization, shape (d_model,)
        path: Path to pull back through, a list of tuples of the form (component_name, layer, head)
        mlp_out: Whether to use MLP outputs or activations

    Returns:
        A dict of feature vectors and de-embeddings:
        {
            "feature_vectors": (n_components + 1)-length list of (d_model,) tensors
            "deembeddings": (n_components + 1)-length list of (vocab_size,) tensors
        }
    """
    vecs = [start_vector]  # Always do direct path

    # Get cache
    layers = [component[1] for component in path if component[0] == "mlp"]
    _, cache = model.run_with_cache(example, names_filter=[utils.get_act_name("resid_mid", layer) for layer in layers])

    while path:
        component = path.pop(0)
        if component[0] == "attention":
            vecs.append(vecs[-1] @ model.OV[component[1]][component[2]])
        elif component[0] == "mlp":
            mid_acts = cache[utils.get_act_name("resid_mid", component[1])]
            x_mid = mid_acts[0, token_idx][None, None, :]
            my_fun = ln2_mlp_until_out if mlp_out else ln2_mlp_until_post
            vecs.append(
                get_tangent_plane_at_point(
                    x_mid,
                    lambda x: my_fun(x, model.blocks[component[1]].ln2, model.blocks[component[1]].mlp, use_ln=False),
                    vecs[-1],
                )[0, 0]
            )

    deembeddings = [model.W_E @ vec for vec in vecs]

    return {"feature_vectors": vecs, "deembeddings": deembeddings}
