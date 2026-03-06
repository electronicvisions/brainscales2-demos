'''
Helpers for the tutorial: tutorial_11-genetic_algorithms_mc.
'''
from __future__ import annotations
from typing import List, Sequence
from dataclasses import dataclass

import ipywidgets as widget
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import PathCollection
from matplotlib.axes import Axes


def plot_membrane_traces(membrane_traces: List, compartment_chain) -> None:
    """
    Display recorded membrane traces.

    Creates a grid of plots of size 2 by `compartment_chain.length`. In the
    first row the recorded membrane trace of each compartment is shown. The
    compartment membership is thereby indicated by the column.
    In the second row the recorded membrane traces are shown with subtracted
    baseline.

    :param membrane_traces: List of recorded membrane traces of the different
        compartments.
    :param compartment_chain: Compartment chain class to obtain spike times
        as well as the chain length from.
    """
    fig, axes = plt.subplots(2, compartment_chain.length, sharex=True,
                             sharey='row', figsize=(10, 4))

    def plot_traces(axes: np.ndarray, subtract_baseline: bool = False) -> None:
        """
        Plot each compartment's membrane trace in a subplot.
        If subtract_baseline is true the baseline is subtracted.

        :param axes: Axes to plot the traces into. For each compartment a
            single axes must be provided.
        :param subtract_baseline: Bool deciding to subtract the baseline
            potential.
        """
        for measured in range(compartment_chain.length):
            membrane_trace = membrane_traces[measured]
            input_time = compartment_chain.spike_time

            sliced_trace = membrane_trace.time_slice(
                t_start=input_time - 0.01 * pq.ms,
                t_stop=input_time + 0.06 * pq.ms)

            # Normalize voltage and times
            sliced_trace.times = (sliced_trace.times - input_time).rescale(
                pq.us)
            if subtract_baseline:
                sliced_trace = sliced_trace - sliced_trace[:100].mean()

            axes[measured].plot(sliced_trace.times, sliced_trace, c='C0')

            # Indicate amplitudes of EPSP
            axes[measured].axhline(np.max(sliced_trace), c='C3', ls='--')
            axes[measured].axhline(sliced_trace[:100].mean(), c='C3', ls='--')

    for axis in axes.flatten():
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

    for axis in axes[:, 1:].flatten():
        axis.axes.get_yaxis().set_visible(False)
        axis.spines['left'].set_visible(False)

    for n_comp, axis in enumerate(axes[0, :]):
        axis.set_title(f'Compartment {n_comp}')

    axes[0, 0].set_ylabel('Membrane Voltage\n(MADC)')
    axes[1, 0].set_ylabel('Baseline subtracted\nMembrane Voltage\n(MADC)')

    plot_traces(axes[0])
    plot_traces(axes[1], subtract_baseline=True)

    fig.supxlabel('Hardware Time (us)')
    fig.suptitle(r'$\triangleleft$ recording site $\triangleright$', y=1.02)


@dataclass
class PlotArchive:
    """
    Store data relevant for plotting older configurations of the chip.
    """
    scatter_data: np.ndarray
    line_data: np.ndarray
    popt: np.ndarray
    pcov: np.ndarray
    g_leak: int
    g_ic: int


def scatter_plot(axes: Axes, x_values: Sequence, y_values: Sequence, **kwargs):
    '''
    Update scatter plot.
    '''
    return axes.scatter(x_values, y_values, **kwargs).get_offsets().data


def line_plot(axes: Axes, x_values: Sequence, y_values: Sequence, *,
              popt: np.ndarray, pcov: np.ndarray, g_leak: int, g_ic: int,
              **kwargs):
    '''
    Update line plot and label.
    '''
    label = rf'$\lambda_{{\mathrm{{emp}}}}=({popt[0]:.2f}\pm$' \
        + rf'{np.sqrt(pcov[0][0]):.2f}$)$ ' \
        + rf'$g_{{\mathrm{{leak}}}}=${g_leak}, ' \
        + rf'$g_{{\mathrm{{ic}}}}=${g_ic}'
    return axes.plot(x_values, y_values, label=label, **kwargs)[0].get_data()


def visualize_experiment(
        popt: np.ndarray, pcov: np.ndarray, amplitudes: np.ndarray,
        compartment_chain, old_data: List[PlotArchive]) -> None:
    """
    Visualize the attenuation experiment.

    The amplitudes of the EPSPs are visualized in a scatter plot and a curve
    resembling the fit of an exponential is added to the corresponding data.
    The supplied amplitudes and conductances are stored in `old_data`.

    :param popt: Fit parameters from curve_fit.
    :param pcov: Covariance matrix of the fit parameters.
    :param amplitudes: Amplitudes of the EPSPs in each compartment.
    :param compartment_chain: Compartment chain class to obtain current
        conductance configuration as well as the chain length from.
    :param old_data: List to save old runs into.
    """
    _, axes = plt.subplots(figsize=(10, 6))
    # only integer valued x-ticks
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))

    scat_data = scatter_plot(axes, range(compartment_chain.length), amplitudes,
                             color='C3', zorder=10)
    # fit_length_constant norms the amplitudes to the largest amplitude found
    # rescale the fit with that amplitude
    x_values = np.linspace(0, compartment_chain.length - 1, 1000)
    line_data = line_plot(
        axes, x_values, amplitudes[0] * compartment_chain.fitfunc(
            x_values, *popt), popt=popt, pcov=pcov,
        g_leak=compartment_chain.g_leak, g_ic=compartment_chain.g_ic,
        color='C3', zorder=10)
    old_data.append(PlotArchive(
        scat_data, line_data, popt, pcov, compartment_chain.g_leak,
        compartment_chain.g_ic))
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes.set_xlabel("Compartment")
    axes.set_ylabel("EPSP Amplitude (MADC)")
    axes.set_ylim(-1, 350)


def display_old_runs(archive: List[PlotArchive], n_last_runs: int):
    '''
    Display the `n_last_runs` recorded attenuations provided by `archive`.

    :param archive: Archive to plot old traces from.
    :param n_last_runs: Number of archived data to plot.
    '''
    _, axes = plt.subplots(figsize=(10, 6))
    # only integer valued x-ticks
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    # display older runs
    for n_old in range(min(len(archive), n_last_runs)):
        data = archive[-(n_old + 1)]
        kwargs = {'color': f'C{n_old + 4}', 'zorder': n_last_runs - n_old,
                  'alpha': 1.8 / (2 + n_old)}
        line_plot(axes, *data.line_data, popt=data.popt, pcov=data.pcov,
                  g_leak=data.g_leak, g_ic=data.g_ic, **kwargs)
    axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes.set_xlabel("Compartment")
    axes.set_ylabel("EPSP Amplitude (MADC)")
    axes.set_ylim(-1, 350)


def plot_grid_search(fig: plt.Figure, axes: np.ndarray, lambdas: np.ndarray,
                     conductances: np.ndarray) -> None:
    '''
    Visualize the grid search results.

    The length constant from a grid search is visualized in a two dimensional
    grid where the x-axis expresses the leak conductance and the y-axis the
    inter-compartment conductance. The length constants are color coded
    according to a color bar beneath the plot in the second axis.

    :param fig: Figure to plot axes into.
    :param axes: Axes to display the grid search and the color bar beneath.
    :param lambdas: Results of a grid search.
    :param conductances: The used conductances to obtain the results from.
    '''
    g_leak, g_ic = np.meshgrid(conductances, conductances)
    img = axes[0].pcolormesh(g_leak, g_ic, lambdas)

    cbar = fig.colorbar(img, cax=axes[1], orientation='horizontal')
    cbar.set_label(r"$\lambda_{\mathrm{emp}}$ [compartments]")
    axes[0].set_xlabel(r"$g_{\mathrm{leak}}$")
    axes[0].set_ylabel(r"$g_{\mathrm{ic}}$")


def plot_fitness_population(
        axes: Axes, popsize: int, std_target: float) -> None:
    '''
    Creates `popsize` + 3 empty line plots on the provided axes.

    One line plot for each individual's fitness, one for the mean fitness,
    one for the min fitness and one to indicate the trial-to-trial
    variation.
    Additionally the labels and limits are set.

    :param axes: Axes to plot line plots into.
    :param popsize: Number of empty line plots (-1) that will be created.
    '''
    for _ in range(popsize):
        axes.plot([], [], color='gray', alpha=0.2)
    axes.plot([], [], color='black', label="mean population")
    axes.plot([], [], color='blue', label="best individual")
    axes.axhline(std_target, ls='--', color="black",
                 label="trial-to-trial boundary")
    axes.legend()
    axes.set_xlabel("generation")
    axes.set_ylabel(r"fitness (deviation from target) $\hat{{\lambda}}$")
    axes.set_xlim(0, 10)
    axes.set_ylim(1e-4, 4)
    axes.set_yscale('log')


def update_figure(population: Sequence, fig: plt.Figure,
                  scat: PathCollection) -> None:
    """
    Update figure with current population and the fitness progress.

    :param population: Population containing individuals. The individuals will
                        be plotted on the grid and their fitness will be
                        illustrated next to it.
    :param fig: Figure that will be updated.
    :param scat: Scatter plot that will be updated.
    """
    # update current population in scatter plot
    scat.set_offsets(population)
    # update fitness of visualization
    arr_fitness = []
    for n_line, line in enumerate(fig.axes[1].get_lines()[:-3]):
        data = line.get_data()
        fitness = data[1]

        # no data yet
        if len(fitness) == 0:
            fitness = np.array([population[n_line].fitness.values[0]])
        else:
            fitness = np.vstack(
                (fitness, population[n_line].fitness.values[0]))
        line.set_data(range(len(fitness)), fitness)
        arr_fitness.append(fitness)
    # mean fitness
    fig.axes[1].get_lines()[-3].set_data(
        range(len(fitness)), np.mean(np.asarray(arr_fitness), axis=0))
    # min fitness
    fig.axes[1].get_lines()[-2].set_data(
        range(len(fitness)), np.min(np.asarray(arr_fitness), axis=0))
    display(fig)  # pylint: disable=undefined-variable


def visualization(population: Sequence, fig: plt.Figure,
                  scat: PathCollection, output: widget.Output) -> str:
    """
    Clears the previous output and updates the figure.

    :param population: Population containing individuals.
    :param fig: Figure that will be updated.
    :param scat: Scatter plot that will be updated.
    :param output: IPython Output which will be manipulated.
    :return: String which is printed with deap's statistics tool.
    """
    output.clear_output(wait=True)
    with output:
        update_figure(population, fig, scat)
    return ""
