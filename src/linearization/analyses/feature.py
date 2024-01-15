import torch

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from typing import Dict

import tqdm

from ..sae_tutorial import AutoEncoder
from ..vars import SAE_CFG


def _get_sample(model, sae, data, feature_idx, layer, act_name, num_batches):
    all_activations = []
    all_tokens = []
    with torch.no_grad():
        for i in tqdm.trange(num_batches):
            tokens = data[torch.randperm(len(data))[: SAE_CFG["model_batch_size"]]]

            _, cache = model.run_with_cache(tokens, names_filter=get_act_name(act_name, layer))
            mlp_acts = cache[get_act_name(act_name, layer)]

            hidden = sae(mlp_acts)[2]

            all_activations.append(hidden[:, :, feature_idx])
            all_tokens.append(tokens)

        activations = torch.cat(all_activations, dim=0)
        tokens = torch.cat(all_tokens, dim=0)

    return activations, tokens


def top_activating_examples(
    model: HookedTransformer,
    sae: AutoEncoder,
    data: torch.Tensor,
    feature_idx: int,
    layer,
    act_name: str = "post",
    num_batches: int = 25,
    num_examples: int = 10,
    reverse: bool = False,
    uniform: bool = False,
    rank: bool = False,
) -> Dict[str, torch.Tensor]:
    activations, tokens = _get_sample(model, sae, data, feature_idx, layer, act_name, num_batches)

    if uniform and rank:
        idx = torch.argsort(activations.flatten(), descending=True)
        idx = idx[activations.flatten()[idx] > 0]
        spacing = max(1, len(idx) // num_examples)
        idx = idx[::spacing][:num_examples]
    elif uniform:
        max_activation = torch.max(activations.flatten())
        activation_targets = torch.linspace(0, max_activation, num_examples).to(activations.device)
        idx = torch.argmin(torch.abs(activations.flatten()[:, None] - activation_targets[None, :]), dim=0)
    else:
        idx = torch.topk(activations.flatten(), num_examples, dim=0, largest=not reverse).indices

    n = activations.shape[1]
    r, c = idx // n, idx % n

    return {"examples": tokens[r], "activations": activations[r], "rows": r, "cols": c}


def top_logit_tokens(model, sae, data, feature_idx, num_examples=10, reverse=False):
    raise NotImplementedError
