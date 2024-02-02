# Taken from https://github.com/HoagyC/sparse_coding/blob/69c5ae06813ee77bafa679dfacee33b21395d8e4/autoencoders/learned_dict.py

from abc import ABC, abstractmethod
from typing import Optional

import torch

# from torch import nn
from torchtyping import TensorType

from typing import Tuple

# from autoencoders.ensemble import DictSignature

# _n_dict_components, _activation_size, _batch_size = (None, None, None)  # type: Tuple[None, None, None]


class LearnedDict(ABC):
    n_feats: int
    activation_size: int

    @abstractmethod
    def get_learned_dict(self) -> TensorType["_n_dict_components", "_activation_size"]:
        pass

    @abstractmethod
    def encode(
        self, batch: TensorType["_batch_size", "_activation_size"]
    ) -> TensorType["_batch_size", "_n_dict_components"]:
        pass

    @abstractmethod
    def to_device(self, device):
        pass

    def decode(
        self, code: TensorType["_batch_size", "_n_dict_components"]
    ) -> TensorType["_batch_size", "_activation_size"]:
        learned_dict = self.get_learned_dict()
        x_hat = torch.einsum("nd,bn->bd", learned_dict, code)
        return x_hat

    def center(
        self, batch: TensorType["_batch_size", "_activation_size"]
    ) -> TensorType["_batch_size", "_activation_size"]:
        # overloadable method to center the batch for the (otherwise) linear model
        return batch

    def uncenter(
        self, batch: TensorType["_batch_size", "_activation_size"]
    ) -> TensorType["_batch_size", "_activation_size"]:
        # inverse of `center`
        return batch

    def predict(
        self, batch: TensorType["_batch_size", "_activation_size"]
    ) -> TensorType["_batch_size", "_activation_size"]:
        batch_centered = self.center(batch)
        c = self.encode(batch_centered)
        x_hat_centered = self.decode(c)
        x_hat = self.uncenter(x_hat_centered)
        return x_hat

    def n_dict_components(self):
        return self.get_learned_dict().shape[0]


class TiedSAE(LearnedDict):
    def __init__(self, encoder, encoder_bias, centering=(None, None, None), norm_encoder=True):
        self.encoder = encoder
        self.encoder_bias = encoder_bias
        self.norm_encoder = norm_encoder
        self.n_feats, self.activation_size = self.encoder.shape

        center_trans, center_rot, center_scale = centering

        if center_trans is None:
            center_trans = torch.zeros(self.activation_size)

        if center_rot is None:
            center_rot = torch.eye(self.activation_size)
            print(center_rot)

        if center_scale is None:
            center_scale = torch.ones(self.activation_size)

        self.center_trans = center_trans
        self.center_rot = center_rot
        self.center_scale = center_scale

    def initialize_missing(self):
        if not hasattr(self, "center_trans"):
            self.center_trans = torch.zeros(self.activation_size, device=self.encoder.device)

        if not hasattr(self, "center_rot"):
            self.center_rot = torch.eye(self.activation_size, device=self.encoder.device)

        if not hasattr(self, "center_scale"):
            self.center_scale = torch.ones(self.activation_size, device=self.encoder.device)

    def center(self, batch):
        return (
            torch.einsum("cu,bu->bc", self.center_rot, batch - self.center_trans[None, :]) * self.center_scale[None, :]
        )

    def uncenter(self, batch):
        return (
            torch.einsum("cu,bc->bu", self.center_rot, batch / self.center_scale[None, :]) + self.center_trans[None, :]
        )

    def get_learned_dict(self):
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, 1e-8)[:, None]

    def to_device(self, device):
        self.initialize_missing()

        self.encoder = self.encoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)

        self.center_trans = self.center_trans.to(device)
        self.center_rot = self.center_rot.to(device)
        self.center_scale = self.center_scale.to(device)

    def encode(self, batch):
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder

        c = torch.einsum("nd,bd->bn", encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c
