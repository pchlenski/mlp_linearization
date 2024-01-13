def top_activating_examples(model, data, sae, feature_idx, num_examples=10, reverse=False):
    raise NotImplementedError


def uniform_examples(model, data, sae, feature_idx, num_examples=10, rank=False):
    raise NotImplementedError


def top_logit_tokens(model, data, sae, feature_idx, num_examples=10, reverse=False):
    raise NotImplementedError


def uniform_logit_tokens(model, data, sae, feature_idx, num_examples=10, rank=False):
    raise NotImplementedError
