import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_loss_accuracy(train_loss, validation_loss, batch):
#def plot_loss_accuracy(train_loss, batch):
    epochs = len(train_loss)
    fig, (ax1) = plt.subplots(1)
    ax1.plot(list(range(epochs)), train_loss, label='Training Loss')
    ax1.plot(list(range(epochs)), validation_loss, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Epoch vs Loss (batch %s)' %batch)
    ax1.legend()



def plot_dataloader_distribution(dataloader, label_names):
    n_labels = len(label_names)
    batch_distribution = {l: [] for l in range(n_labels)}

    for _, b in dataloader:
        for l in range(n_labels):
            batch_distribution[l].append(int((b == l).sum()))

    x = np.arange(len(batch_distribution[0]))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, freq in batch_distribution.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, freq, width, label=label_names[label])
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Batch index')
    ax.set_ylabel('Number of samples')
    ax.set_title('Dataloader class distribution')
    ax.set_xticks(x + width, x)
    ax.legend(loc='upper left')

    plt.show()
