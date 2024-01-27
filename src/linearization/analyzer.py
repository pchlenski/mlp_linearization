import torch

from typing import List, Union

from .loading import load_model, load_data, load_sae
from .vars import MODEL, RUN, DATASET

from .analyses.model import frequencies, f1_scores
from .analyses.feature import top_activating_examples, top_logit_tokens
from .analyses.example import attributions
from .analyses.path import feature_vectors


class SAELinearizer:
    def __init__(
        self,
        model_name: str = MODEL,
        dataset_name: str = DATASET,
        sae_names: List[str] = [RUN],
        layers: List[int] = [0],
        act_name: str = "post",
        device: str = None,
        **kwargs,
    ):
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.act_name = act_name

        self.set_model(
            model_name=model_name,
            dataset_name=dataset_name,
            sae_names=sae_names,
            layers=layers,
            act_name=act_name,
            **kwargs,
        )

    def set_model(
        self,
        model_name: str,
        dataset_name: str,
        sae_names: List[str],
        layers: List[int],
        act_name: str,
        seed=42,
        run_analysis=True,
        num_batches=25,
        **kwargs,
    ):
        # Load model, data, and SAE(s)
        self.model = load_model(model_name, **kwargs).to(self.device)
        self.data = load_data(self.model, dataset_name, **kwargs).to(self.device)
        self.saes = {sae_name: load_sae(sae_name, **kwargs).to(self.device) for sae_name in sae_names}
        self.sae_layers = dict(zip(sae_names, layers))
        self.act_name = act_name
        self.seed = seed
        self._kw1 = {"model": self.model, "data": self.data, "act_name": self.act_name}

        # Run analysis
        if run_analysis:
            torch.manual_seed(self.seed)
            self.frequencies = {
                name: frequencies(
                    **self._kw1, sae=self.saes[name], layer=self.sae_layers[name], num_batches=num_batches
                )
                for name in self.saes
            }
            self.f1_scores = {
                name: f1_scores(**self._kw1, sae=self.saes[name], layer=self.sae_layers[name], num_batches=num_batches)
                for name in self.saes
            }

        # Unset downstream values
        self._clean("model")

    def set_feature(self, feature_idx: int, sae_name: str = None, run_analysis=True, num_batches=25):
        # Single-SAE case + no SAE name provided
        if sae_name is None and len(self.saes) == 1:
            sae_name = list(self.saes.keys())[0]
        elif sae_name is None:
            raise ValueError("Must provide SAE name when multiple SAES are loaded")

        # Set feature and feature vector
        self.sae = self.saes[sae_name]
        self.layer = self.sae_layers[sae_name]
        self.feature_idx = feature_idx
        # if self.act_name == "mlp_out":
        #     self.feature_vector = self.sae.W_enc[:, feature_idx]
        # elif self.act_name == "post":
        #     self.feature_vector = self.sae.W_enc[:, feature_idx] @ self.model.blocks[self.layer].mlp.W_out
        self.feature_vector = self.sae.W_enc[:, feature_idx]
        self._kw2 = {**self._kw1, "sae": self.sae, "feature_idx": self.feature_idx, "layer": self.layer}

        # Run analysis
        if run_analysis:
            torch.manual_seed(self.seed)

            # SAE level
            self.top_examples = top_activating_examples(**self._kw2, num_batches=num_batches)
            self.bottom_examples = top_activating_examples(**self._kw2, num_batches=num_batches, reverse=True)
            self.uniform_examples = top_activating_examples(**self._kw2, num_batches=num_batches, uniform=True)
            self.uniform_ranked_examples = top_activating_examples(
                **self._kw2, num_batches=num_batches, uniform=True, rank=True
            )

            # Logit level
            self.top_logit_tokens = top_logit_tokens(**self._kw2)
            self.bottom_logit_tokens = top_logit_tokens(**self._kw2, reverse=True)

        # Unset downstream values
        self._clean("feature")

    def set_example(self, example: Union[str, int], token_idx: int, run_analysis=True):
        # Turn prompt or index into equivalent tensors
        if isinstance(example, str):
            example = self.model.tokenizer.encode(example)
            example = torch.tensor(example)
            print(example.shape)

            # Deal with <BOS> and <EOS> tokens
            bos = self.model.tokenizer.bos_token_id
            eos = self.model.tokenizer.eos_token_id
            pad = self.model.tokenizer.pad_token_id
            if example[0] != bos:
                example = torch.cat([torch.tensor([bos]), example])
            if example[-1] != eos:
                example = torch.cat([example, torch.tensor([eos])])
            # if len(example) != self.data.shape[1]:
            #     n_pad = self.data.shape[1] - len(example)
            #     example = torch.cat([example, torch.tensor([pad] * n_pad)])

        elif isinstance(example, int):
            example = self.data[example]

        else:
            raise ValueError("Example must be a string or integer")

        # Set example and token index
        self.example = example
        self.token_idx = token_idx
        print(f"Token: {self.model.tokenizer.decode(self.example[self.token_idx])}")  # Sanity check token idx
        self._kw3 = {**self._kw2, "example": self.example, "token_idx": self.token_idx}

        # Run analysis
        if run_analysis:
            torch.manual_seed(self.seed)
            # self.attributions = attributions(**self._kw3)
            self.attributions = attributions(
                self.model,
                self.feature_vector,
                self.example,
                self.token_idx,
                self.layer,
                mlp_out=self.act_name == "mlp_out",
            )

        # Unset downstream values
        self._clean("example")

    def set_path(self, path: List[str], run_analysis=True):
        # Intake tuples: [("attention", layer, head), ("mlp", layer), etc.]
        # Should be ordered from deepest to shallowest nodes of computation graph
        # Input validation and cleaning (use proper component names)
        path_names_fixed = []
        for component in path:
            if isinstance(component, tuple):
                if component[0] == "attention":
                    path_names_fixed.append(("attention", component[1], component[2]))
                elif component[0] == "mlp":
                    path_names_fixed.append(("mlp", component[1]))
                else:
                    raise ValueError("Invalid component name")
            else:
                raise ValueError("Invalid component type: need tuple")

        # Set attributes
        self.path = path_names_fixed
        self._kw4 = {**self._kw3, "start_vector": self.feature_vector, "path": self.path}

        # Run analysis
        if run_analysis:
            torch.manual_seed(self.seed)

            self.feature_vectors = feature_vectors(**self._kw4)
            # self.deembeddings = deembeddings(**self._kw4)

    def _clean(self, component: str):
        # Wipe path-level attributes
        if component in ["model", "feature", "example"]:
            for attr in ["path", "_kw4", "feature_vectors", "deembeddings"]:
                if hasattr(self, attr):
                    delattr(self, attr)

        # Wipe example-level attributes
        if component in ["model", "feature"]:
            for attr in ["example", "token_idx", "_kw3", "attributions"]:
                if hasattr(self, attr):
                    delattr(self, attr)

        # Wipe feature-level attributes
        if component == "model":
            for attr in [
                "sae",
                "layer",
                "feature_idx",
                "feature_vector",
                "_kw2",
                "top_examples",
                "bottom_examples",
                "uniform_examples",
                "uniform_ranked_examples",
                "top_logit_tokens",
                "bottom_logit_tokens",
                "uniform_logit_tokens",
                "uniform_ranked_logit_tokens",
            ]:
                if hasattr(self, attr):
                    delattr(self, attr)
