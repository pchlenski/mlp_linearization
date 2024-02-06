import numpy as np
import matplotlib.pyplot as plt


def fix_token(model, token):
    token_str = model.tokenizer.decode([token], clean_up_tokenization_spaces=False)
    # Escape special characters to ensure text is treated as verbatim
    token_str_escaped = token_str.translate(
        str.maketrans(
            {
                "_": r"\_",
                "^": r"\^",
                "$": r"\$",
                "{": r"\{",
                "}": r"\}",
                "\\": r"\\",
                "~": r"\textasciitilde{}",
                "&": r"\&",
                "%": r"\%",
            }
        )
    )
    return token_str_escaped


def visualize_topk(examples, activations, columns, model, trim=10, zero_negatives=False):
    # Get sizes and pad
    n_examples, seq_length = examples.shape
    pad_width = ((0, 0), (trim, trim))
    padded_examples = np.pad(examples, pad_width, mode="constant", constant_values=model.tokenizer.pad_token_id)
    padded_activations = np.pad(activations, pad_width, mode="constant", constant_values=0)

    # Prepare arrays for trimmed and padded data
    trimmed_examples = np.zeros((n_examples, trim * 2 + 1), dtype=padded_examples.dtype)
    trimmed_activations = np.zeros((n_examples, trim * 2 + 1), dtype=padded_activations.dtype)

    for i, col in enumerate(columns):
        # Adjust column index for padding
        col += trim
        # Extract slices for this example
        trimmed_examples[i] = padded_examples[i, col - trim : col + trim + 1]
        trimmed_activations[i] = padded_activations[i, col - trim : col + trim + 1]

    # Set up plot
    fig = plt.figure(figsize=(trimmed_examples.shape[1], trimmed_examples.shape[0] // 2))
    ax = fig.add_subplot(111)

    # Clean up examples and activations
    if zero_negatives:
        trimmed_activations[trimmed_activations < 0] = 0

    cax = ax.imshow(trimmed_activations, cmap="coolwarm")  # , vmin=0)
    fig.colorbar(cax, ax=ax, orientation="vertical")

    # for i, row in examples.iterrows():
    # for i, (_, row) in enumerate(examples.iterrows()):  # Hacky but I need indices
    for i, row in enumerate(trimmed_examples):
        for j, token in enumerate(row):
            # if token not in ["<|EOS|>", "<|PAD|>", "<|BOS|>"]:
            if token not in [0, 1, model.tokenizer.eos_token, model.tokenizer.pad_token, model.tokenizer.bos_token]:
                plt.text(j, i, fix_token(model, token), ha="center", va="center", fontsize=8)

    return ax
