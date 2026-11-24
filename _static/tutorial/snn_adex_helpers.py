"""
Helpers for the tutorial: adex_complex_dynamics.
"""

import warnings
from collections import OrderedDict
from functools import partial

import numpy as np
import quantities

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from IPython.display import display
from ipywidgets import interactive, IntSlider, Layout, VBox, Box, HTML

plt.style.use("_static/matplotlibrc")

IntSlider = partial(IntSlider, continuous_update=False)


# pylint: disable=invalid-name, too-many-locals, too-many-statements
def plot_membrane_dynamics(pop):
    """
    Plot membrane trace, adaptation state, and the phase space trajectory of a
    two-neuron population `pop` comprising one target and one readout neuron.
    """

    w_snippets = pop.get_data("adaptation").segments[0].\
        irregularlysampledsignals
    t_w = w_snippets[0].times
    w = np.array(w_snippets[0])
    for ws in w_snippets[1:]:
        t_w = np.concatenate((t_w, ws.times))
        w = np.concatenate((w, np.array(ws)))

    v_snippets = pop.get_data("v").segments[0].irregularlysampledsignals
    t_v = v_snippets[0].times
    v = np.array(v_snippets[0])
    for vs in v_snippets[1:]:
        t_v = np.concatenate((t_v, vs.times))
        v = np.concatenate((v, np.array(vs)))

    # we enable the current pulse in the second experiment snippet
    # -> spiketrains[1]
    spike_times = pop[0:1].get_data().segments[-1].spiketrains[1]

    fig = plt.figure(figsize=(11, 4))
    grid = gs.GridSpec(2, 2,
                       width_ratios=[1.5, 1], height_ratios=[1.5, 1],
                       wspace=0.2)

    ax_v = fig.add_subplot(grid[0, 0])
    ax_w = fig.add_subplot(grid[1, 0])
    ax_p = fig.add_subplot(grid[:, 1:])

    ax_v.set_xlim(round(t_v.min() * 10) / 10, round(t_v.max() * 10) / 10)
    ax_v.set_ylim(150, 750)

    ax_w.set_xlim(round(t_w.min() * 10) / 10, round(t_w.max() * 10) / 10)
    ax_w.set_ylim(150, 750)

    ax_p.set_xlim(150, 750)
    ax_p.set_ylim(150, 750)

    ax_v.set_xticklabels([])
    ax_v.set_ylabel("membrane potential")

    ax_w.set_xlabel("time / s")
    ax_w.set_ylabel("adaptation state")

    ax_p.set_xlabel("membrane potential")
    ax_p.set_ylabel("adaptation state")

    # check for saturating voltages and raise warning
    lower_limit = 200
    upper_limit = 680
    fraction_v_lower_limit = (v < lower_limit).sum() / v.size
    fraction_v_upper_limit = (v > upper_limit).sum() / v.size

    fraction_w_lower_limit = (w < lower_limit).sum() / w.size
    fraction_w_upper_limit = (w > upper_limit).sum() / w.size

    if fraction_v_lower_limit > 0.2:
        warnings.warn("The membrane is close to the lower voltage limit."
                      "Increase the leak potential!")
        ax_v.axhspan(v.min(), lower_limit, fc="r", ec="r", alpha=0.1)
    if fraction_v_upper_limit > 0.1:
        warnings.warn("The membrane is close to the upper voltage limit."
                      "Reduce the leak potential!")
        ax_v.axhspan(upper_limit, v.max(), fc="r", ec="r", alpha=0.1)
    if fraction_w_lower_limit > 0.2:
        warnings.warn("The adaptation trace is below the lower voltage limit."
                      "Increase the adaptation reference potential!")
        ax_w.axhspan(w.min(), lower_limit, fc="r", ec="r", alpha=0.1)
    if fraction_w_upper_limit > 0.1:
        warnings.warn("The adaptation trace is above the upper voltage limit."
                      "Reduce the adaptation reference potential!")
        ax_w.axhspan(upper_limit, w.max(), fc="r", ec="r", alpha=0.1)

    n_samples = min(v.size, w.size)

    # cut (mask out) downswing after action potential from trace
    # to clean up phase trajectory plot
    spike_mask = np.zeros(n_samples, dtype=bool)
    t = np.arange(n_samples) * np.mean(np.diff(t_v)) * quantities.ms
    for spike_time in spike_times:
        mask_start = spike_time - 0.1 * quantities.us
        mask_close = spike_time + 1.5 * quantities.us
        spike_mask[(t > mask_start) & (t < mask_close)] = 1

    ax_v.plot(t_v, v, c="k")
    ax_w.plot(t_w, w, c="k")

    v_n = v[:n_samples].astype(float)
    w_n = w[:n_samples].astype(float)

    # fill downswings with None, appear in plot as gap
    v_n[spike_mask] = None
    w_n[spike_mask] = None

    # plot gray trajectory including post-spike jumps
    # by slicing based on mask defined above
    ax_p.plot(v_n[~spike_mask], w[:n_samples][~spike_mask], c="lightgray")

    # plot trace with gaps
    ax_p.plot(v_n, w_n, c="k")


layout = {"width": "calc(100% - 10px)"}
style = {"description_width": "120px"}

headers = {
    "config": "Configuration",
    "leak": "Leak and reset",
    "adaptation": "Adaptation",
    "stimulus": "Stimulus",
    "exponential": "Exponential"
}

# dict of neuron parameters available in slider GUI with the following data:
# parameter_name: (label, (lower_limit, upper_limit), default, section)
# note: only first 126 neurons (of this quadrant) can be used, as one
# circuit is required for readout dummy
# pylint: disable=line-too-long
neuron_parameters = OrderedDict([
    ("target_neuron", ("target neuron", (0, 126), 0, "config")),
    ("leak_v_leak", ("leak potential", (400, 1000), 700, "leak")),
    ("leak_i_bias", ("leak bias", (20, 500), 70, "leak")),
    ("reset_v_reset", ("reset potential", (400, 1000), 600, "leak")),
    ("adaptation_i_bias_tau", ("time constant bias", (20, 600), 100, "adaptation")),
    ("adaptation_i_bias_a", ("a bias", (0, 300), 100, "adaptation")),
    ("adaptation_i_bias_b", ("b bias", (0, 500), 0, "adaptation")),
    ("adaptation_v_ref", ("baseline potential", (200, 1000), 500, "adaptation")),
    ("exponential_v_exp", ("onset potential", (300, 1000), 900, "exponential")),
    ("exponential_i_bias", ("slope bias", (100, 1000), 300, "exponential")),
    ("constant_current_i_offset", ("stimulus current", (0, 1000), 300, "stimulus"))
])
# pylint: enable=line-too-long

last_configuration = {}


def build_gui(callback,
              controls,
              config=None,
              defaults=None,
              copy_configuration=False):
    """
    Build a slider-based GUI for an experiment callback function.
    The sliders are grouped by the specific circuit they affect and the
    callback's result (e.g. graph) is displayed above the sliders.
    """

    if config is None:
        config = {}
    if defaults is None:
        defaults = {}

    # instantiate sliders according to list of parameters provided by the user
    sliders = OrderedDict()
    for c in controls:
        spec = neuron_parameters[c]
        default = defaults[c] if c in defaults else spec[2]
        if copy_configuration and c in last_configuration:
            default = last_configuration[c]
        sliders[c] = IntSlider(min=spec[1][0],
                               max=spec[1][1],
                               step=1,
                               value=default,
                               description=spec[0],
                               layout=layout,
                               style=style)

    widget = interactive(callback, **sliders, **config)

    # group sliders according to their sections
    sections = OrderedDict()
    for c, slider in sliders.items():
        header = neuron_parameters[c][3]
        if header not in sections:
            sections[header] = OrderedDict()
        sections[header][c] = slider

    # build UI according to hierarchical structure from above
    ui = []
    for header, children in sections.items():
        ui.append([])

        ui[-1].append(HTML(f"<h3>{headers[header]}</h3>"))

        for slider in children.values():
            ui[-1].append(slider)

    output = widget.children[-1]

    # define custom layout following the responsive web design paradigm
    slider_box = Box(tuple(VBox(tuple(s)) for s in ui), layout=Layout(
        display='grid',
        grid_template_columns='repeat(auto-fit, minmax(400px, 1fr))',
        width='100%'
    ))

    display(VBox([output, slider_box]))

    widget.update()
