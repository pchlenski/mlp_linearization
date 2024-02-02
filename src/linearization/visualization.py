import matplotlib.pyplot as plt


def visualize_topk(examples, activations, columns, trim=10):
    # Set up plot
    fig = plt.figure(figsize=(examples.shape[1] // 2, examples.shape[0] // 2))
    ax = fig.add_subplot(111)

    # Clean up examples and activations
    activations[activations < 0] = 0
    # if trim:
    #     activations = activations[:, columns - trim : columns + trim]
    # That didn't work, but trim it this way:
    if trim:
        indices = [i for column in columns for i in range(column - trim, column + trim + 1)]
        activations = activations[:, indices]
        examples = examples[:, indices]
    ax.matshow(activations, cmap="coolwarm", vmin=0)

    # for i, row in examples.iterrows():
    # for i, (_, row) in enumerate(examples.iterrows()):  # Hacky but I need indices
    for i, row in enumerate(examples):
        for j, token in enumerate(row):
            if token not in ["<|EOS|>", "<|PAD|>", "<|BOS|>"]:
                plt.text(j, i, token, ha="center", va="center", fontsize=8)

    # ax.set_xticks(range(len(examples.columns)), examples.columns)
    # ax.set_yticks(
    #     range(len(examples.index)), [f"R={examplesi}, C={ci}" for examplesi, ci in zip(examples.index, columns)]
    # )

    return ax
