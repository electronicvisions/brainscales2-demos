# pylint: skip-file
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import random
import ipywidgets as w


def show_transforms(data, Rotation, encode):

    fig = plt.figure(figsize=(7 * 3, 5))
    idx = random.randint(0, len(data))

    img, label = data[idx]

    ax_1 = fig.add_subplot(1, 3, 1)
    divider = make_axes_locatable(ax_1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax_1.matshow(img[0])
    ax_1.set_title("image of a " + str(label))
    fig.colorbar(im, cax=cax)
    ax_1.set_xlabel("pixel")
    ax_1.set_ylabel("pixel")

    ax_2 = fig.add_subplot(1, 3, 2)
    divider = make_axes_locatable(ax_2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax_2.matshow(Rotation(img)[0])
    ax_2.set_title("randomly rotated " + str(label))
    fig.colorbar(im, cax=cax)
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


def plot_training(epochs: int):

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
        n1 = len(all_data["losses_train"])
        n2 = len(all_data["accs_train"])
        n3 = len(all_data["rates_train"])

        if n1 >= 1 and n2 >= 1 and n3 >= 1:
            loss_line_train.set_data(
                np.arange(1, n1 + 1), all_data["losses_train"])
            accs_line_train.set_data(
                np.arange(1, n2 + 1), all_data["accs_train"])
            rates_line.set_data(np.arange(1, n3 + 1), all_data["rates_train"])

            loss_line_test.set_data(
                np.arange(1, n1 + 1), all_data["losses_test"])
            accs_line_test.set_data(
                np.arange(1, n2 + 1), all_data["accs_test"])
            rates_line_test.set_data(
                np.arange(1, n3 + 1), all_data["rates_test"])

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
