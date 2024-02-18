import torch
import tqdm

from collections import Counter
from typing import Dict, List

# from scipy.sparse import csr_matrix
import transformer_lens
from transformer_lens.utils import get_act_name

from ..sae_tutorial import get_freqs, AutoEncoder
from ..vars import SAE_CFG


def frequencies(
    model: transformer_lens.HookedTransformer,
    sae: AutoEncoder,
    data: torch.Tensor,
    layer: int,
    num_batches: int = 25,
    act_name: str = "post",
) -> torch.Tensor:
    """
    Given a model, data, and SAE, return a tensor of frequencies for each SAE feature. Also prints number of dead
    features.

    Note that this is a sample over a random subset of the data, so the frequencies are not exact.

    Args:
        model: A HookedTransformer model from transformer_lens.
        sae: A trained sparse autoencoder.
        data: A tensor of data with shape (n_samples, n_tokens).
        layer: The layer whose activations the SAE is trained on.
        num_batches: The number of batches to sample.
        act_name: The name of the activation to analyze: "post" or "mid"

    Returns:
        A tensor of activation frequencies with shape (n_features,).
    """
    return get_freqs(
        model=model, local_encoder=sae, all_tokens=data, layer=layer, num_batches=num_batches, act_name=act_name
    )


def f1_scores(
    model: transformer_lens.HookedTransformer,
    sae: AutoEncoder,
    data: torch.Tensor,
    layer: int,
    num_batches: int = 25,
    act_name: str = "post",
) -> Dict[str, List[float]]:
    """
    Given a model, data, and SAE, return a dict of precision, recall, and f1 scores for each SAE feature.

    Note that this is a sample over a random subset of the data, so the frequencies are not exact.

    Args:
        model: A HookedTransformer model from transformer_lens.
        sae: A trained sparse autoencoder.
        data: A tensor of data with shape (n_samples, n_tokens).
        layer: The layer whose activations the SAE is trained on.
        num_batches: The number of batches to sample.
        act_name: The name of the activation to analyze: "post" or "mid"

    Returns:
        A dict of precision, recall, and f1 scores for each SAE feature, each of which is a list of floats.
    """
    feature_counters = [Counter() for _ in range(sae.W_enc.shape[1])]
    token_counter = Counter()

    # Count up activating tokens and total tokens for our features
    with torch.no_grad():
        for i in tqdm.trange(num_batches):
            tokens = data[torch.randperm(len(data))[: SAE_CFG["model_batch_size"]]]

            layer_name = "ln2" if act_name == "normalized" else None
            _, cache = model.run_with_cache(tokens, names_filter=get_act_name(act_name, layer, layer_name))
            mlp_acts = cache[get_act_name(act_name, layer, layer_name)]

            hidden = sae(mlp_acts)[2]
            for j in range(sae.W_enc.shape[1]):
                activating_indices = torch.argwhere(hidden[:, :, j] > 0)
                activating_tokens = tokens[activating_indices[:, 0], activating_indices[:, 1]]

                # Update feature counters with activating tokens
                feature_counters[j].update(activating_tokens.cpu().numpy())

            # Update token counter with ALL token occurrences
            token_counter.update(tokens.reshape(-1).cpu().numpy())

    # Get f1 scores for each feature:
    precisions = []
    recalls = []
    f1_scores = []
    top_tokens = []
    for feature_counter in feature_counters:
        # try:
        if feature_counter:
            top_token_id, top_token_count = feature_counter.most_common(1)[0]
            top_token = model.tokenizer.decode(top_token_id)
            precision = top_token_count / sum(feature_counter.values())
            recall = top_token_count / token_counter[top_token_id]
            f1 = 2 * precision * recall / (precision + recall)
        else:
            precision, recall, f1, top_token = 0, 0, 0, None
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        top_tokens.append(top_token)

    return {"precisions": precisions, "recalls": recalls, "f1_scores": f1_scores, "top_tokens": top_tokens}
