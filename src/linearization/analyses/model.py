import torch
import tqdm

from collections import Counter

# from scipy.sparse import csr_matrix
from transformer_lens.utils import get_act_name

from ..sae_tutorial import get_freqs
from ..vars import SAE_CFG


def frequencies(model, sae, data, layer, num_batches=25, act_name="post") -> torch.Tensor:
    return get_freqs(
        model=model, local_encoder=sae, all_tokens=data, layer=layer, num_batches=num_batches, act_name=act_name
    )


def f1_scores(model, sae, data, layer, num_batches=25, act_name="post") -> dict:
    feature_counters = [Counter() for _ in range(sae.W_enc.shape[1])]
    token_counter = Counter()

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
