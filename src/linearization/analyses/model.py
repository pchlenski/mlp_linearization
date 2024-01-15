import torch
import tqdm

from collections import Counter
from typing import Dict, List

# from scipy.sparse import csr_matrix
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

from ..sae_tutorial import get_freqs, AutoEncoder
from ..vars import SAE_CFG


def frequencies(
    model: HookedTransformer,
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
    model: HookedTransformer,
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

            _, cache = model.run_with_cache(tokens, names_filter=get_act_name(act_name, layer))
            mlp_acts = cache[get_act_name(act_name, layer)]

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
    for feature_counter in feature_counters:
        try:
            top_token, top_token_count = feature_counter.most_common(1)[0]
            precision = top_token_count / sum(feature_counter.values())
            recall = top_token_count / token_counter[top_token]
            f1 = 2 * precision * recall / (precision + recall)
        except IndexError:  # For dead and low-frequency features
            precision, recall, f1 = 0, 0, 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {"precisions": precisions, "recalls": recalls, "f1_scores": f1_scores}
