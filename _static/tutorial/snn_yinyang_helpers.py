# pylint: skip-file
import matplotlib.pyplot as plt
import numpy as np
import torch


COLORS = ["#6392EB", "#D97C75", "#65AD68"]


def plot_data(example_point, data: torch.Tensor, target: torch.Tensor):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax.scatter(data[target == 0, 0], data[target == 0, 1], color=COLORS[0],
               label="yin")
    ax.scatter(data[target == 1, 0], data[target == 1, 1], color=COLORS[1],
               label="yang")
    ax.scatter(data[target == 2, 0], data[target == 2, 1], color=COLORS[2],
               label="dot")
    ax.scatter(*example_point[:2], marker="D", color="black", label="example")
    ax.set_xlabel(r"x")
    ax.set_ylabel(r"y")
    ax.legend()


def plot_input_encoding(spikes, t_early, t_late, t_bias, t_sim, dt):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.gca()
    ax.scatter(torch.nonzero(spikes)[:, 0] * dt, torch.nonzero(spikes)[:, 1])
    ax.vlines(t_early, -0.5, 4.5, ls=":")
    ax.vlines(t_late, -0.5, 4.5, ls=":")
    ax.vlines(t_bias, -0.5, 4.5, ls=":")
    ax.axvspan(t_early, t_late, facecolor='grey', alpha=0.2)

    ax.set_xlabel(r"t [$\mu s$]")
    ax.set_ylabel(r"y")
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlim(0, t_sim)
    ax.set_yticks(
        [0, 1, 2, 3, 4],
        ["$t^i_0$", "$t^i_1$", "$t^i_2$", "$t^i_3$", r"$t^\mathrm{bias}_4$"])
    ax.set_xticks(
        [0, t_early, t_late, t_bias, t_sim],
        [0, r"$t_\mathrm{early}$", rf"$t_\mathrm{{late}} = {{{t_late}}}$",
         rf"$t_\mathrm{{bias}} = {{{t_bias}}}$", r"$t_\mathrm{sim}$"])


def plot_training(n_hidden, t_sim, dt):

    train_style = {"color": "blue", "alpha": 0.5, "label": "Training"}
    test_style = {"color": "red", "alpha": 1., "marker": "o", "label": "Test"}
    spike_style = {"color": "black", "alpha": 0.7, "marker": "|"}

    all_data = {
        "scores": None,
        "data": None,
        "spikes_in": None,
        "spikes_h": None,
        "trace_out": None,
        "n_batches": None,
        "epochs": [],
        "losses": [],
        "accs": [],
        "rates": [],
        "losses_test": [],
        "accs_test": [],
        "rates_test": []}

    fig = plt.figure(constrained_layout=True, figsize=(7, 12))
    subfigs = fig.subfigures(nrows=3, height_ratios=[8, 2, 2])
    axs = subfigs[0].subplots(nrows=3)

    # Loss
    axs[0].set_xlabel(r"Epoch")
    axs[0].set_ylabel(r"loss")
    loss_line, = axs[0].plot([], [], **train_style)
    loss_line_test, = axs[0].plot([], [], **test_style)
    # Accuracy
    axs[1].set_xlabel(r"Epoch")
    axs[1].set_ylabel(r"Accuracy [%]")
    axs[1].set_yscale("log")
    accs_line, = axs[1].plot([], [], **train_style)
    accs_line_test, = axs[1].plot([], [], **test_style)
    # Firing rate
    axs[2].set_xlabel(r"Epoch")
    axs[2].set_ylabel(r"Firing rate hidden layer")
    rates_line, = axs[2].plot([], [], **train_style)
    rates_line_test, = axs[2].plot([], [], **test_style)

    # Scores
    axs_s = subfigs[1].subplots(ncols=3)

    # Example
    axs_e = subfigs[2].subplots(ncols=3)
    axs_e[0].set_ylim(-0.5, 4.5)
    axs_e[0].set_ylabel("Inputs")
    sc1 = axs_e[0].scatter(0, 0, **spike_style)
    axs_e[1].set_ylim(-0.5, n_hidden - 0.5)
    axs_e[1].set_ylabel("Hidden")
    sc2 = axs_e[1].scatter(0, 0, **spike_style)
    axs_e[2].set_ylabel("y [a.u]")
    for i in range(3):
        axs_e[i].set_xlim(0, t_sim)
        axs_e[i].set_xlabel(r"$t$ [us]")
    example_lines = []
    for i in range(3):
        line, = axs_e[2].plot([], [], color=COLORS[i])
        example_lines.append(line)

    def update():
        if len(all_data["losses"]):
            epochs_train = 1. / all_data["n_batches"] * \
                np.arange(0, len(all_data["losses"]))
            loss_line.set_data(epochs_train, all_data["losses"])
            accs_line.set_data(epochs_train, all_data["accs"])
            rates_line.set_data(epochs_train, all_data["rates"])
        epochs_test = np.arange(len(all_data["losses_test"]))
        loss_line_test.set_data(epochs_test, all_data["losses_test"])
        accs_line_test.set_data(epochs_test, all_data["accs_test"])
        rates_line_test.set_data(epochs_test, all_data["rates_test"])

        for i in range(3):
            alphas = all_data["scores"][:, i] / all_data["scores"].sum(1)
            # PyTorch does the same
            alphas[all_data["scores"].sum(1) <= 0.] = 1. if i == 0 else 0.
            axs_s[i].clear()
            axs_s[i].scatter(
                all_data["data"][:, 0], all_data["data"][:, 1], linewidths=0,
                s=10, color=COLORS[i], alpha=alphas)
            axs_s[i].set_xlabel(r"$x$")
            axs_s[i].set_ylabel(r"$y$")

        # example
        sc1.set_offsets(
            np.c_[torch.nonzero(all_data["spikes_in"][:, 0])[:, 0].numpy() * dt,
                  torch.nonzero(all_data["spikes_in"][:, 0])[:, 1].numpy()])
        sc2.set_offsets(
            np.c_[torch.nonzero(all_data["spikes_h"][:, 0])[:, 0].numpy() * dt,
                  torch.nonzero(all_data["spikes_h"][:, 0])[:, 1].numpy()])
        for i, line in enumerate(example_lines):
            line.set_data(
                np.arange(all_data["trace_out"][:, 0].shape[0]) * dt,
                all_data["trace_out"][:, 0, i])

        for i in range(3):
            axs[i].relim()
            axs[i].autoscale_view()
            axs[i].legend()

        for i in range(3):
            axs_e[i].relim()
            axs_e[i].autoscale_view()

        display(fig)

    def update_train(n_batches, loss, acc, rate):
        all_data["losses"].append(loss)
        all_data["accs"].append(acc)
        all_data["rates"].append(rate)
        all_data["n_batches"] = n_batches

    def update_test(spikes_in, spikes_h, trace_out, data, target,
                    scores, loss, acc, rate):
        all_data["losses_test"].append(loss)
        all_data["accs_test"].append(acc)
        all_data["rates_test"].append(rate)
        all_data["scores"] = scores
        all_data["data"] = data
        all_data["spikes_in"] = spikes_in
        all_data["spikes_h"] = spikes_h
        all_data["trace_out"] = trace_out

    return update, update_train, update_test
