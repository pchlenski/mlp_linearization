#! /home/phil/miniconda3/envs/mats_sae_training/bin/python

import torch
import os
import sys

sys.path.append("/home/phil/transcoders")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    is_transcoder=True,
    model_name="gelu-1l",
    # model_name="gpt2-small",
    # hook_point=f"blocks.{i}.ln2.hook_normalized",
    hook_point="blocks.0.ln2.hook_normalized",
    out_hook_point="blocks.0.hook_mlp_out",
    hook_point_layer=0,
    out_hook_point_layer=0,
    # hook_point_layer=i,
    # d_in=768,
    d_in=512,
    d_out=512,
    # dataset_path="Skylion007/openwebtext",
    dataset_path="NeelNanda/c4-code-20k",
    is_dataset_tokenized=False,
    # SAE Parameters
    expansion_factor=32,
    b_dec_init_method="geometric_median",
    # Training Parameters
    lr=0.0004,
    l1_coefficient=0.00008,
    lr_scheduler_name="constantwithwarmup",
    train_batch_size=4096,
    context_size=128,
    lr_warm_up_steps=5000,
    # Activation Store Parameters
    n_batches_in_buffer=128,
    total_training_tokens=1_000_000 * 1000,
    # total_training_tokens=1_000_000 * 300,
    store_batch_size=32,
    # Dead Neurons and Sparsity
    use_ghost_grads=False,
    # use_ghost_grads=True,
    # feature_sampling_method=None,
    # feature_sampling_window=1000,
    # dead_feature_window=5000,
    # dead_feature_threshold=1e-6,
    # WANDB
    log_to_wandb=True,
    # log_to_wandb=False,
    # wandb_project="mats_sae_training_gpt2",
    wandb_project="gelu1l_transcoder_training",
    wandb_entity="chlenski",
    wandb_log_frequency=100,
    # Misc
    device="cuda",
    seed=42,
    # n_checkpoints=10,
    # checkpoint_path="checkpoints",
    dtype=torch.float32,
)

sparse_autoencoder = language_model_sae_runner(cfg)
