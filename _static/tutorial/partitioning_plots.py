"""
Helper functions for plotting training progress and data transformations
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import display


def show_transforms(data, rotation, encode):
    """
    Visualizes transformations and encodings of an image from the given
    dataset.

    This function generates a figure with three subplots:
    1. The original image from the dataset.
    2. The image after applying a random rotation transformation.
    3. The sparse encoding of the image.

    :param data: A dataset containing tuples of images and their corresponding
        labels.
    :param Rotation: A function that applies a random rotation transformation
        to an image.
    :param encode: A function that encodes an image into sparse spike events.

    :return: A matplotlib figure containing the visualizations.
    """

    fig = plt.figure(figsize=(7 * 3, 5))
    idx = random.randint(0, len(data))

    img, label = data[idx]

    ax_1 = fig.add_subplot(1, 3, 1)
    divider = make_axes_locatable(ax_1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    mat = ax_1.matshow(img[0])
    ax_1.set_title("image of a " + str(label))
    fig.colorbar(mat, cax=cax)
    ax_1.set_xlabel("pixel")
    ax_1.set_ylabel("pixel")

    ax_2 = fig.add_subplot(1, 3, 2)
    divider = make_axes_locatable(ax_2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    mat = ax_2.matshow(rotation(img)[0])
    ax_2.set_title("randomly rotated " + str(label))
    fig.colorbar(mat, cax=cax)
    ax_2.set_xlabel("pixel")
    ax_2.set_ylabel("pixel")

    ax_3 = fig.add_subplot(1, 3, 3)
    enc_ind = encode(img).reshape(30, 28**2).to_sparse().coalesce().indices()
    ax_3.scatter(
        enc_ind[0], enc_ind[1], marker="|", color="black", linewidths=1,
        label="spike events")
    ax_3.set_xlabel(r"time [$\mu$s]")
    ax_3.set_ylabel("neuron index")
    ax_3.set_title("encoding of the " + str(label))
    ax_3.legend()

    return fig


# pylint: disable=too-many-locals, too-many-statements
def plot_training(epochs: int):
    """
    Creates a set of plots to visualize training and testing metrics over
    epochs.

    This function generates three subplots for accuracy, loss, and firing rate,
    and provides update functions to dynamically add data to the plots during
    training and testing.

    :param epochs: The total number of epochs for which the plots will be
        created. This determines the x-axis range for the plots.

    :return: A tuple containing three functions:
             - `update`: A function to refresh the plots with the latest data.
             - `update_train`: A function to append training data (loss,
                accuracy, rate).
             - `update_test`: A function to append testing data (loss,
                accuracy, rate).
    """

    train_style = {"color": "teal", "label": "Training"}
    test_style = {"color": "orangered", "label": "Test"}

    all_data = {
        "losses_train": [],
        "accs_train": [],
        "rates_train": [],
        "losses_test": [],
        "accs_test": [],
        "rates_test": []}

    fig = plt.figure(constrained_layout=True, figsize=(7, 12))
    subfigs = fig.subfigures(nrows=1)
    axs = subfigs.subplots(nrows=3)

    # Accuracy
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy [%]")
    axs[0].set_ylim(0.0, 1.00)
    axs[0].set_xlim(0.5, epochs + 0.5)
    axs[0].grid(which='major', color='#CCCCCC', linewidth=0.8)
    axs[0].grid(which='minor', color='#EEEEEE', linewidth=0.5)
    axs[0].minorticks_on()
    accs_line_train, = axs[0].plot([], [], **train_style)
    accs_line_test, = axs[0].plot([], [], **test_style)

    # Loss
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].set_ylim(0, 2.5)
    axs[1].set_xlim(0.5, epochs + 0.5)
    axs[1].grid(which='major', color='#CCCCCC', linewidth=0.8)
    axs[1].grid(which='minor', color='#EEEEEE', linewidth=0.5)
    axs[1].minorticks_on()
    loss_line_train, = axs[1].plot([], [], **train_style)
    loss_line_test, = axs[1].plot([], [], **test_style)

    # Firing rate
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Firing rate [spikes/input/neuron]")
    axs[2].set_ylim(0, 1.5)
    axs[2].set_xlim(0.5, epochs + 0.5)
    axs[2].grid(which='major', color='#CCCCCC', linewidth=0.8)
    axs[2].grid(which='minor', color='#EEEEEE', linewidth=0.5)
    axs[2].minorticks_on()
    rates_line, = axs[2].plot([], [], **train_style)
    rates_line_test, = axs[2].plot([], [], **test_style)

    def update():
        n_loss = len(all_data["losses_train"])
        n_acc = len(all_data["accs_train"])
        n_rate = len(all_data["rates_train"])

        if n_loss >= 1 and n_acc >= 1 and n_rate >= 1:
            loss_line_train.set_data(
                np.arange(1, n_loss + 1), all_data["losses_train"])
            accs_line_train.set_data(
                np.arange(1, n_acc + 1), all_data["accs_train"])
            rates_line.set_data(
                np.arange(1, n_rate + 1), all_data["rates_train"])

            loss_line_test.set_data(
                np.arange(1, n_loss + 1), all_data["losses_test"])
            accs_line_test.set_data(
                np.arange(1, n_acc + 1), all_data["accs_test"])
            rates_line_test.set_data(
                np.arange(1, n_loss + 1), all_data["rates_test"])

        for i in range(3):
            axs[i].legend()

        display(fig)

    def update_train(loss, acc, rate):
        all_data["losses_train"].append(loss)
        all_data["accs_train"].append(acc)
        all_data["rates_train"].append(rate)

    def update_test(loss, acc, rate):
        all_data["losses_test"].append(loss)
        all_data["accs_test"].append(acc)
        all_data["rates_test"].append(rate)

    return update, update_train, update_test
