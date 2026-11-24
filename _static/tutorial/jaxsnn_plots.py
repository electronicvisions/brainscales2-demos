import numpy as np
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from matplotlib import gridspec


def plot_training_progress(loss, times, weights_arr):
    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot loss
    axes[0].plot(loss, label='Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True)
    axes[0].legend()

    # Plot times
    axes[1].plot(times, label='Spike Time')
    axes[1].axhline(0.015, color='r', linestyle='--', label='Target Time')
    axes[1].set_title('Spike Time over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Spike Time')
    axes[1].set_yscale('log')
    axes[1].grid(True)
    axes[1].legend()

    # Plot weights (one trace for each weight)
    for i in range(weights_arr.shape[1]):
        axes[2].plot(weights_arr[:, i], label=f'Weight {i}')
    axes[2].set_title('Weights over Epochs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Weight Value')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def plot_trainset_samples(data, targets, random_sample):
    """
    Plot the trainset samples with a random sample highlighted.

    :param data: Array of data points.
    :param targets: Array of target labels.
    :param random_sample: A tuple containing a random data point and its label.
    """
    plt.figure(figsize=(6, 6))  # Make the figure square
    plt.scatter(
        data[:, 0], data[:, 1], c=targets, cmap='viridis', alpha=0.7,
        edgecolors='none',
    )

    # Mark the random sample with a red star
    plt.scatter(
        random_sample[0][0], random_sample[0][1], color='red', marker='*',
        s=200,
    )

    # Add legend for class colors
    classes = ['Class 0', 'Class 1', 'Class 2']
    colors = [plt.cm.viridis(i / 2) for i in range(3)]  # pylint: disable=no-member
    for class_label, color in zip(classes, colors):
        plt.scatter([], [], c=[color], label=class_label, alpha=0.7)

    plt.legend(title='Classes')
    plt.title('Visualization of Trainset Samples')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.axis('equal')  # Ensure the plot is square
    plt.grid(True)
    plt.show()


# pylint: disable=too-many-arguments
def plot_encoding(
    encoded_spikes, encoded_targets, targets, random_index, t_late, t_max,
):
    """
    Plot the input and target encoding in a single figure with two rows.

    :param encoded_spikes: Encoded spike times for the input sample.
    :param encoded_targets: Encoded target spike times.
    :param t_late: Time indicating the late spike threshold.
    :param t_max: Maximum time for the x-axis.
    """
    _, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot input encoding
    axes[0].set_title("Input Encoding")
    for neuron_idx, spike_times in enumerate(encoded_spikes.time[0][::5]):
        color = 'blue' if neuron_idx > 0 else 'red'  # Bias spike in red
        axes[0].scatter(spike_times, neuron_idx, color=color, alpha=0.7)
    axes[0].axvline(
        x=t_late, color='black', linestyle='dotted',
        label=r"$t_\mathrm{late}$",
    )
    axes[0].axvline(
        x=t_max, color='gray', linestyle='dotted', label=r"$t_\mathrm{max}$",
    )
    axes[0].set_ylabel("Neuron Index")
    axes[0].set_yticks(range(len(encoded_spikes.time[0][::5])))
    axes[0].legend()
    axes[0].grid(True)

    # Plot target encoding
    axes[1].set_title("Target Encoding")
    for neuron_idx, spike_times in enumerate(encoded_targets[0]):
        # Highlight the correct output neuron in a different color
        color = 'blue'
        if neuron_idx == np.argmax(targets[random_index]):
            color = 'green'
        axes[1].scatter(spike_times, neuron_idx, color=color, alpha=0.7)
    axes[1].axvline(
        x=t_late, color='black', linestyle='dotted',
        label=r"$t_\mathrm{late}$",
    )
    axes[1].axvline(
        x=t_max, color='gray', linestyle='dotted', label=r"$t_\mathrm{max}$",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Neuron Index")
    axes[1].set_yticks(range(len(encoded_targets[0])))
    axes[1].legend()
    axes[1].grid(True)

    # Set x-axis limits for both plots
    axes[1].set_xlim(-t_max * 0.1, t_max * 1.1)

    plt.tight_layout()
    plt.show()


def create_plot(epochs, xy_testset, test_samples, output_size):
    # pylint: disable=too-many-locals, too-many-statements
    losses = []
    error_rates = []
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(10, 8))

    # Define the grid structure (2 rows, 6 columns)
    gspec = gridspec.GridSpec(2, 6, figure=fig)

    ax1 = fig.add_subplot(gspec[0, 0:3])
    ax1.set_title("Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale('log')
    ax1.set_xlim(0, epochs)
    ax1.grid(True)

    ax2 = fig.add_subplot(gspec[0, 3:6])
    ax2.set_title("Test Error Rate")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Error Rate")
    ax2.set_xlim(0, epochs)
    ax2.set_yscale('log')
    ax2.grid(True)

    # Add subplots for the second row (3 columns)
    ax3 = fig.add_subplot(gspec[1, 0:2])
    ax4 = fig.add_subplot(gspec[1, 2:4])
    ax5 = fig.add_subplot(gspec[1, 4:6])
    axes = [ax3, ax4, ax5]
    # Set up loss plot
    line_loss, = ax1.plot([], [], 'o-', color="blue", markersize=4)

    # Set up error rate plot
    line_err, = ax2.plot([], [], 'o-', color="red", markersize=4)

    plt.tight_layout()
    plt.ion()  # Turn on interactive mode
    display(fig)

    # Original x, y coordinates from the unencoded testset
    xy_coords = np.array(xy_testset[0][:, :2]).reshape(test_samples, 2)

    def update_plots(loss, error_rate, t_first_spike):
        """Update both plots with current loss and error rate data."""
        # pylint: disable=too-many-locals
        losses.append(float(loss))  # Convert JAX array to Python float
        error_rates.append(float(error_rate))  # Convert JAX array to float
        x_data = list(range(len(losses)))

        # Update loss plot line
        line_loss.set_data(x_data, losses)

        # Update error rate plot line
        line_err.set_data(x_data, error_rates)

        # Update y-axis limits to fit the data
        if len(losses) > 0:
            # For loss plot - log scale needs positive values
            loss_min = max(min(losses) * 0.8, 1e-5)
            loss_max = max(losses) * 1.2
            ax1.set_ylim(loss_min, loss_max)

            # For error rate plot - log scale needs positive values
            err_min = max(min(error_rates) * 0.8, 1e-5)
            err_max = max(error_rates) * 1.2
            ax2.set_ylim(err_min, err_max)

        scores = np.array(t_first_spike).reshape(test_samples, output_size)
        scores = np.where(np.isfinite(scores), scores, np.nan)
        finite = scores[np.isfinite(scores)]
        if finite.size > 0:
            vmin = np.min(finite)
            vmax = np.percentile(finite, 95)
        else:
            vmin, vmax = 0.0, 1.0
        cmap = plt.cm.viridis.copy()  # pylint: disable=no-member
        cmap.set_bad("lightgray")

        for i in range(output_size):
            ax = axes[i]
            ax.clear()  # avoid stacking scatters from previous epochs
            _ = ax.scatter(
                xy_coords[:, 0], xy_coords[:, 1], c=scores[:, i], cmap=cmap,
                vmin=vmin, vmax=vmax, alpha=0.8, edgecolors='none', s=10,
                plotnonfinite=True,
            )

            ax.set_ylabel('y')
            ax.set_xlabel('x')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, linestyle='--', alpha=0.5)

        # Redraw the figure
        fig.canvas.draw()
        plt.pause(0.001)  # Small pause to ensure plot updates
        clear_output(wait=True)
        display(fig)

    return update_plots
