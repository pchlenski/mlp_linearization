from typing import List

from .loading import load_model, load_data, load_sae
from .analyses.model import frequencies, f1_scores
from .analyses.feature import top_activating_examples, uniform_examples, top_logit_tokens, uniform_logit_tokens
from .analyses.example import attributions
from .analyses.path import feature_vectors, deembeddings


class SAELinearizer:
    def __init__(self, model_name: str, dataset_name: str, sae_names: List[str], device: str = None, **kwargs):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load model, data, and SAE(s)
        self.model = load_model(model_name, **kwargs).to(self.device)
        self.data = load_data(self.model, dataset_name, **kwargs).to(self.device)
        self.saes = {sae_name: load_sae(sae_name, **kwargs).to(self.device) for sae_name in sae_names}
        self._kw1 = {"model": self.model, "data": self.data}

        # Run analysis
        self.frequencies = {name: frequencies(**_kw1, sae=self.saes[name]) for name in self.saes}
        self.f1_scores = {name: f1_scores(**_kw1, sae=self.saes[name]) for name in self.saes}

        # Unset downstream values
        del self.sae, self.feature_idx, self.feature_vector, self.sample, self.token_idx, self.path
        del self.frequencies, self.f1_scores
        del self.top_examples, self.bottom_examples, self.uniform_examples, self.uniform_ranked_examples

    def set_feature(self, sae_name: str, feature_idx: int):
        # Set feature and feature vector
        self.sae = self.saes[sae_name]
        self.feature_idx = feature_idx
        self.feature_vector = self.saes[sae_name][:, feature_idx]
        self._kw2 = {**kw1, "sae": self.sae, "feature_idx": self.feature_idx}

        # SAE analysis
        self.top_examples = top_activating_examples(**kw2)
        self.bottom_examples = top_activating_examples(**kw2, reverse=True)
        self.uniform_examples = uniform_examples(**kw2)
        self.uniform_ranked_examples = uniform_examples(**kw2, rank=True)

        # Logit weights
        self.top_logit_tokens = top_logit_tokens(**kw2)
        self.bottom_logit_tokens = top_logit_tokens(**kw2, reverse=True)
        self.uniform_logit_tokens = uniform_logit_tokens(**kw2)
        self.uniform_ranked_logit_tokens = uniform_logit_tokens(**kw2, rank=True)

        # Unset downstream values
        del self.sample, self.token_idx, self.path

    def set_example(self, token_idx: int, example_idx: int = None, prompt: str = None):
        # Ensure we have something to work with
        if example_idx is None and prompt is None:
            raise ValueError("Must provide either example_idx or prompt")
        elif example_idx is not None and prompt is not None:
            raise ValueError("Cannot provide both example_idx and prompt")

        # Turn prompt or index into equivalent tensors
        if prompt is not None:
            sample = self.model.tokenizer(prompt).to(self.device)  # TODO: verify this does <BOS> <EOS> stuff or fix
        else:
            sample = self.data[example_idx]

        # Set example and token index
        self.sample = sample
        self.token_idx = token_idx
        self._kw3 = {**kw2, "sample": self.sample, "token_idx": self.token_idx}

        # Run analysis
        self.attributions = attributions(**kw3)

        # Unset downstream values
        del self.path

    def set_path(self, path: List[str]):
        self.path = path
        self._kw4 = {**kw3, "path": self.path}

        # Run analysis
        self.feature_vectors = feature_vectors(**kw4, path=self.path)
        self.deembeddings = deembeddings(**kw4, path=self.path)
