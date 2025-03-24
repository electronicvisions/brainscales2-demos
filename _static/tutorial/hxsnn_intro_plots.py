# pylint: skip-file
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import ipywidgets as w
import torch


def plot_training(inputs, target, epochs):
    input_events = torch.nonzero(inputs)
    fig, axs = plt.subplots(3, constrained_layout=True)
    losses = []
    colors = ["red", "blue", "green"]
    pred_lines = [axs[2].plot([], [], color=c)[0] for c in colors]
    axs[1].scatter(input_events[:, 0], input_events[:, 2], s=1, color="black")
    for i in range(3):
        axs[2].plot(target[:, 0, i], color=colors[i], ls="--")
    axs[2].sharex(axs[1])
    axs[2].set_xlabel(r"Time [$\mu$s]")
    axs[2].set_ylabel("$v_m$ [a.u.]")
    axs[1].set_xlim(0, 100)
    axs[1].set_ylabel("Input Neuron")
    axs[0].set_ylim(0.0001, 1)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("MSE")
    axs[0].set_xlim(0, epochs)
    loss_line, = axs[0].plot([0], [1], color="blue")
    axs[0].set_yscale('log')

    def update_plot(loss, y):
        losses.append(loss)
        loss_line.set_data(np.arange(len(losses)), losses)
        for i in range(3):
            pred_lines[i].set_data(
                np.arange(0, 100), y.membrane_cadc.detach().numpy()[:, 0, i])
        display(fig)
    return update_plot


def plot_compare_traces(inputs, z):
    input_events = torch.nonzero(inputs)
    output_events = torch.nonzero(z.spikes)

    _, axs = plt.subplots(nrows=2, sharex='col')
    axs[0].vlines(input_events[:, 0], ymin=-100, ymax=0, color="orange", label="Inputs")
    axs[0].vlines(output_events[:, 0], ymin=-100, ymax=0, color="red", label="Outputs")
    axs[0].plot(z.membrane_cadc.detach().numpy().reshape(-1), color="blue")
    axs[0].set_ylim(-60, 0)
    axs[0].set_ylabel(r"$v_m^\mathrm{CADC}$ [CADC Value]")
    axs[0].legend()
    axs[1].plot(
        z.membrane_madc[0, :].detach().numpy().reshape(-1) / 125,
        z.membrane_madc[1, :].detach().numpy().reshape(-1), color="blue")
    axs[1].vlines(input_events[:, 0], ymin=250, ymax=550, color="orange")
    axs[1].vlines(output_events[:, 0], ymin=250, ymax=550, color="red")
    axs[1].set_ylabel(r"$v_m^\mathrm{MADC}$  [MADC Value]")
    axs[1].set_ylim(250, 550)
    axs[1].scatter(output_events[:, 0], output_events[:, 2])
    axs[1].set_xlabel(r"time [$\mu$s]")


def plot_mock(inputs, z):
    input_events = torch.nonzero(inputs)
    output_events = torch.nonzero(z.spikes)
    _, axs = plt.subplots(nrows=1, sharex='col', figsize=(5, 3))
    axs.vlines(input_events[:, 0], ymin=60, ymax=140, color="orange", label="Inputs")
    axs.vlines(output_events[:, 0], ymin=60, ymax=140, color="red", label="Outputs")
    axs.plot(z.membrane_cadc.detach().numpy().reshape(-1), color="blue")
    axs.set_ylim(60, 140)
    axs.set_ylabel(r"$v_m^\mathrm{CADC}$ [CADC Value]")
    axs.set_xlabel(r"time [$\mu$s]")
    axs.legend()


def plot_scaled_trace(inputs, bss2_traces, mock_trace):
    _, axs = plt.subplots(nrows=1, sharex='col', figsize=(6, 3))
    input_events = torch.nonzero(inputs)
    axs.vlines(input_events[:, 0], ymin=-1, ymax=2, color="orange", label="Inputs")
    axs.set_ylabel(r"$v_m^\mathrm{CADC}$ [CADC Value]")
    axs.plot(mock_trace, color="red")
    axs.plot(np.stack(bss2_traces).T, color="blue", alpha=0.2)
    axs.plot(np.stack(bss2_traces).mean(0), color="blue")
    axs.set_ylim(-0.2, 0.6)


def plot_targets(targets):
    colors = ["red", "blue", "green"]
    _, ax = plt.subplots(figsize=(6, 3))
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(r"time [$\mu$s]")
    ax.set_ylabel("y [a.u.]")
    for i in range(3):
        ax.plot(targets[:, 0, i].numpy(), color=colors[i])
