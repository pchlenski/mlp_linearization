from typing import List

from .loading import load_model, load_data, load_sae
from .analyses.model import *
from .analyses.feature import *
from .analyses.example import *
from .analyses.path import *


class SAELinearizer:
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        sae_names: List[str],
        device: str = None,
        analyze: bool = True,
        **kwargs
    ):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load model, data, and SAE(s)
        self.model = load_model(model_name, **kwargs).to(self.device)
        self.data = load_data(self.model, dataset_name, **kwargs).to(self.device)
        self.saes = {sae_name: load_sae(sae_name, **kwargs).to(self.device) for sae_name in sae_names}

        # Run analysis
        if analyze:
            self.analyze_model()

        # Unset downstream values
        del self.feature_idx, self.feature_vector, self.sample, self.token_idx, self.path

    def analyze_model(self):
        raise NotImplementedError

    def set_feature(self, sae_name, feature_idx, analyze=True):
        # Set feature and feature vector
        self.feature_idx = feature_idx
        self.feature_vector = self.saes[sae_name][:, feature_idx]

        # Run analysis
        if analyze:
            self.analyze_feature()

        # Unset downstream values
        del self.sample, self.token_idx, self.path

    def analyze_feature(self):
        raise NotImplementedError

    def set_example(self, token_idx, example_idx=None, prompt=None, analyze=True):
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

        # Run analysis
        if analyze:
            self.analyze_example()

        # Unset downstream values
        del self.path

    def analyze_example(self):
        raise NotImplementedError

    def set_path(self, path, analyze=True):
        self.path = path

        # Run analysis
        if analyze:
            self.analyze_path()

    def analyze_path(self):
        raise NotImplementedError
