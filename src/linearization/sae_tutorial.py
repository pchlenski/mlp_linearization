"""
This submodule is a pile of code from Neel's sparse autoencoder tutorial:

https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn?usp=sharing#scrollTo=JUtGmqx8Si-G
"""

import pprint
import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import colorsys
from html import escape
from IPython.display import display
import gradio as gr

from transformer_lens import utils

from functools import partial

from .vars import DTYPES, SPACE, NEWLINE, TAB, SAE_CFG

cfg = SAE_CFG


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # d_hidden = cfg["d_mlp"] * cfg["dict_mult"]
        d_hidden = int(cfg["d_mlp"] * cfg["dict_mult"])  # Because OpenAI's SAE has a weird expansion factor
        d_mlp = cfg["d_mlp"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to("cuda")

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

    # def get_version(self):
    #     return 1+max([int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)])

    # def save(self):
    #     version = self.get_version()
    #     torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
    #     with open(SAVE_DIR/(str(version)+"_cfg.json"), "w") as f:
    #         json.dump(cfg, f)
    #     print("Saved as version", version)

    # def load(cls, version):
    #     cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
    #     pprint.pprint(cfg)
    #     self = cls(cfg=cfg)
    #     self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
    #     return self

    @classmethod
    def load(cls, version):
        if version in ["run1", "run2", "l0", "l1"]:
            return cls.load_from_hf(version)
        elif isinstance(version, int):
            return cls.load_gpt2(version)
        else:
            raise ValueError(f"Unknown version {version}")

    @classmethod
    def load_from_hf(cls, version):
        """
        Loads the saved autoencoder from HuggingFace.

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        if version == "run1":
            hf_id = 25
        elif version == "run2":
            hf_id = 47
        elif version == "l0":
            hf_id = "gelu-2l_L0_16384_mlp_out_51"
        elif version == "l1":
            hf_id = "gelu-2l_L1_16384_mlp_out_50"

        print(f"Loading {version} from HuggingFace at {hf_id}")
        cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{hf_id}_cfg.json")
        if version in ["l0", "l1"]:
            cfg["d_mlp"] = 512  # This was not encoded for some reason
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(
            utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{hf_id}.pt", force_is_torch=True)
        )
        return self

    @classmethod
    def load_gpt2(cls, version):
        # Integer version = gpt2-small layers
        print(f"Loading GPT2-small layer {version} from disk")
        cfg = {"d_mlp": 3072, "dict_mult": 10.6667, "l1_coeff": 0.01, "enc_dtype": "fp32", "seed": 42}
        self = cls(cfg=cfg)
        # self.load_state_dict(torch.load(f"/home/phil/mlp_linearization/data/openai_sae/mlp_post_act/{version}.pt"))
        original_state_dict = torch.load(f"/data_shared/openai_sae/mlp_post_act/{version}.pt")

        # Define a mapping from the original keys to the keys expected by your model
        key_map = {
            "pre_bias": "b_dec",
            "latent_bias": "b_enc",
            "decoder.weight": "W_dec",
            "encoder.weight": "W_enc",
        }

        # Map the keys
        new_state_dict = {key_map.get(k, k): v for k, v in original_state_dict.items()}
        new_state_dict["W_dec"] = new_state_dict["W_dec"].T
        new_state_dict["W_enc"] = new_state_dict["W_enc"].T
        new_state_dict.pop("stats_last_nonzero")

        # Load the new state_dict into your model
        self.load_state_dict(new_state_dict)
        return self


# Utils:
# Get reconstruction loss


def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.0
    return mlp_post


@torch.no_grad()
def get_recons_loss(all_tokens, model, num_batches=5, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    loss_list = []
    for i in range(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[: cfg["model_batch_size"]]]
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(
            tokens,
            return_type="loss",
            fwd_hooks=[(utils.get_act_name("post", 0), partial(replacement_hook, encoder=local_encoder))],
        )
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(
            tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), zero_ablate_hook)]
        )
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(f"loss: {loss:.4f}, recons_loss: {recons_loss:.4f}, zero_abl_loss: {zero_abl_loss:.4f}")
    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)
    print(f"Reconstruction Score: {score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss


# Get frequency
@torch.no_grad()
def get_freqs(all_tokens, model, act_name="post", layer=0, num_batches=25, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).cuda()
    total = 0
    for i in tqdm.trange(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[: cfg["model_batch_size"]]]

        _, cache = model.run_with_cache(tokens, names_filter=utils.get_act_name(act_name, layer))
        mlp_acts = cache[utils.get_act_name(act_name, layer)]
        mlp_acts = mlp_acts.reshape(-1, local_encoder.W_enc.shape[0])

        hidden = local_encoder(mlp_acts)[2]

        act_freq_scores += (hidden > 0).sum(0)
        total += hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores == 0).float().mean()
    print("Num dead", num_dead.item())
    return act_freq_scores


# Visualize feature utils
def create_html(strings, values, max_value=None, saturation=0.5, allow_different_length=False, return_string=False):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [
        s.replace("\n", f"{NEWLINE}<br/>").replace("\t", f"{TAB}&emsp;").replace(" ", "&nbsp;") for s in escaped_strings
    ]

    if isinstance(values, torch.Tensor) and len(values.shape) > 1:
        values = values.flatten().tolist()

    if not allow_different_length:
        assert len(processed_strings) == len(values)

    # scale values
    if max_value is None:
        max_value = max(max(values), -min(values)) + 1e-3
    scaled_values = [v / max_value * saturation for v in values]

    # create html
    html = ""
    for i, s in enumerate(processed_strings):
        if i < len(scaled_values):
            v = scaled_values[i]
        else:
            v = 0
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(hue, v, 1)  # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'
    if return_string:
        return html
    else:
        display(HTML(html))


def basic_feature_vis(text, feature_index, encoder, model, max_val=0):
    feature_in = encoder.W_enc[:, feature_index]
    feature_bias = encoder.b_enc[feature_index]
    _, cache = model.run_with_cache(text, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
    mlp_acts = cache[utils.get_act_name("post", 0)][0]
    feature_acts = F.relu((mlp_acts - encoder.b_dec) @ feature_in + feature_bias)
    if max_val == 0:
        max_val = max(1e-7, feature_acts.max().item())
        # print(max_val)
    # if min_val==0:
    #     min_val = min(-1e-7, feature_acts.min().item())
    return basic_token_vis_make_str(text, feature_acts, model, max_val)


def basic_token_vis_make_str(strings, values, model, max_val=None):
    if not isinstance(strings, list):
        strings = model.to_str_tokens(strings)
    values = utils.to_numpy(values)
    if max_val is None:
        max_val = values.max()
    # if min_val is None:
    #     min_val = values.min()
    header_string = f"<h4>Max Range <b>{values.max():.4f}</b> Min Range: <b>{values.min():.4f}</b></h4>"
    header_string += f"<h4>Set Max Range <b>{max_val:.4f}</b></h4>"
    # values[values>0] = values[values>0]/ma|x_val
    # values[values<0] = values[values<0]/abs(min_val)
    body_string = create_html(strings, values, max_value=max_val, return_string=True)
    return header_string + body_string


# display(HTML(basic_token_vis_make_str(tokens[0, :10], mlp_acts[0, :10, 7], 0.1)))
# # %%
# The `with gr.Blocks() as demo:` syntax just creates a variable called demo containing all these components

# try:
#     demos[0].close()
# except:
#     pass
# demos = [None]


def make_feature_vis_gradio(feature_id, encoder, model, starting_text=None, batch=None, pos=None):
    if starting_text is None:
        starting_text = model.to_string(all_tokens[batch, 1 : pos + 1])
    try:
        demos[0].close()
    except:
        pass
    demos = [None]

    with gr.Blocks() as demo:
        gr.HTML(value=f"Hacky Interactive Neuroscope for gelu-1l")
        # The input elements
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", value=starting_text)
                # Precision=0 makes it an int, otherwise it's a float
                # Value sets the initial default value
                feature_index = gr.Number(label="Feature Index", value=feature_id, precision=0)
                # # If empty, these two map to None
                max_val = gr.Number(label="Max Value", value=None)
                # min_val = gr.Number(label="Min Value", value=None)
                inputs = [text, feature_index, max_val]
        with gr.Row():
            with gr.Column():
                # The output element
                out = gr.HTML(
                    label="Neuron Acts",
                    value=basic_feature_vis(text=starting_text, feature_index=feature_id, encoder=encoder, model=model),
                )
        for inp in inputs:
            inp.change(basic_feature_vis, inputs, out)
    demo.launch(share=True)
    demos[0] = demo


# Inspecting logits


def process_token(s, model):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE + "\n")
    s = s.replace("\t", TAB)
    return s


def process_tokens(l, model):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape) > 1:
        l = l.squeeze(0)
    return [process_token(s, model=model) for s in l]


def process_tokens_index(l, model):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape) > 1:
        l = l.squeeze(0)
    return [f"{process_token(s, model=model)}/{i}" for i, s in enumerate(l)]


def create_vocab_df(logit_vec, model, make_probs=False, full_vocab=None):
    if full_vocab is None:
        full_vocab = process_tokens(model.to_str_tokens(torch.arange(model.cfg.d_vocab)))
    vocab_df = pd.DataFrame({"token": full_vocab, "logit": utils.to_numpy(logit_vec)})
    if make_probs:
        vocab_df["log_prob"] = utils.to_numpy(logit_vec.log_softmax(dim=-1))
        vocab_df["prob"] = utils.to_numpy(logit_vec.softmax(dim=-1))
    return vocab_df.sort_values("logit", ascending=False)


def list_flatten(nested_list):
    return [x for y in nested_list for x in y]


# Make token dataframe


def make_token_df(tokens, model, len_prefix=5, len_suffix=1):
    str_tokens = [process_tokens(model.to_str_tokens(t), model=model) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]

    context = []
    batch = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        # context.append([])
        # batch.append([])
        # pos.append([])
        # label.append([])
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p - len_prefix) : p])
            if p == tokens.shape[1] - 1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p + 1 : min(tokens.shape[1] - 1, p + 1 + len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(
        dict(
            str_tokens=list_flatten(str_tokens),
            unique_token=list_flatten(unique_token),
            context=context,
            batch=batch,
            pos=pos,
            label=label,
        )
    )
