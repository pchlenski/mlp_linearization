# Train a single model
import os
import sys
import torch
import wandb 

sys.path.append("/root/mats_sae_training")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

base_cfg = {
    # Data Generating Function (Model + Training Distibuion)
    "is_transcoder": True,
    "model_name": "gelu-1l",
    "hook_point": "blocks.0.ln2.hook_normalized",
    "out_hook_point": "blocks.0.hook_mlp_out",
    "hook_point_layer": 0,
    "out_hook_point_layer": 0,
    "d_in": 512,
    "d_out": 512,
    "dataset_path": "Skylion007/openwebtext",
    "is_dataset_tokenized": False,
    # SAE Parameters
    "expansion_factor": 128,
    "b_dec_init_method": "geometric_median",
    # Training Parameters
    "lr_scheduler_name": "constantwithwarmup",
    "train_batch_size": 4096,
    "context_size": 128,
    "lr_warm_up_steps": 5000,
    # Activation Store Parameters
    "n_batches_in_buffer": 128,
    "total_training_tokens": 1_000_000 * 300,
    "store_batch_size": 32,
    # Dead Neurons and Sparsity
    "use_ghost_grads": True,
    "feature_sampling_window": 1000,
    "dead_feature_window": 5000,
    "dead_feature_threshold": 1e-6,
    # WANDB
    "log_to_wandb": True,
    "wandb_project": "gelu_1l_sweep",
    "wandb_entity": "chlenski",
    "wandb_log_frequency": 100,
    # Misc
    "device": "cuda",
    "seed": 42,
    "dtype": torch.float32,
}

def train(config=None):
    with wandb.init(config=config):
        wandb_cfg = wandb.config

        base_cfg["lr"] = wandb_cfg.lr
        base_cfg["l1_coefficient"] = wandb_cfg.l1_coefficient
        # base_cfg["lr"] = .001
        # base_cfg["l1_coefficient"] = .00008
        
        language_model_sae_runner(LanguageModelSAERunnerConfig(**base_cfg))

if __name__ == "__main__":
    train()