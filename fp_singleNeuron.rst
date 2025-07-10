Investigating a single neuron
======================================

In this task you can familiarise yourself with the way you will interact with the BrainScaleS-2 chip during the lab exercises.
For all tasks there exist `jupyter` notebooks that are intended to be used as a basis for the results.
If you are not familiar with `python` or `jupyter` notebooks before beginning this lab course, we recommend that you spend some time beforehand to familiarise yourself.

During the execution of some of the notebook cells, you will cause an experiment to be run on a BrainScaleS-2 system remotely.
You will find that due to the analog nature of some of the system's implementation the execution is not deterministic, i.e., not each hardware execution will result in the same outcome.

Generally speaking the simplest hardware experiment can be divided in three steps

- hardware configuration
- hardware execution
- analysis of hardware observables.

More sophisticated experiments can involve multiple iterations of these basic three steps or a subset of them.
At the level of abstraction you will be working on for these lab exercises, most of the intricacies of the hardware configuration and execution will be hidden behind a common high-level API, called `PyNN`.

In the next section we will set up a connection to an RPC server that multiplexes connections to the hardware, which will allow us to work with the system interactively.
This process is handled by a custom microscheduler (*quiggeldy*), a conceptual view of which you can see in the following figure.
The actual hardware execution time has been colored in blue.

.. image:: _static/tutorial/daas_multi.png
    :width: 65%
    :align: center

At the end of this notebook you will find a number of exercises to complete.

Experiment setup
~~~~~~~~~~~~~~~~

.. include:: common_note_helpers.rst

.. only:: jupyter

   To prepare for the execution of experiments on the neuromorphic system, you have to first connect to the scheduling server as follows.
   You will need to run this cell once after starting or resetting the Jupyter backend.
   The easiest way to escape from a broken state, will be to select "Kernel -> Restart" in the menu above and reexecute the following cell.

   .. include:: common_quiggeldy_setup.rst
   .. include:: common_nightly_calibration.rst

   We store results for documentation and to perform evaluations in a common
   place for each of the tasks.

   .. code:: ipython3

      import pathlib
      results_folder = pathlib.Path('results/task1')
      results_folder.mkdir(parents=True, exist_ok=True)


   We start by importing the relevant python packages we need for our experiment and setting up the enviroment for experiment execution.

   .. code:: ipython3

      import pynn_brainscales.brainscales2 as pynn

      from pynn_brainscales.brainscales2 import Population
      from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
      from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse

   The following function allows us to quickly plot the membrane trace and spikes of a neuron.

   .. code:: ipython3

      %matplotlib inline

      import matplotlib.pyplot as plt
      import numpy as np
      import time

      plt.style.use("_static/matplotlibrc")

      def plot_membrane_dynamics(population: Population, segment_id=-1, ylim=None):
          """
          Plot the membrane potential of the neuron in a given population view. Only
          population views of size 1 are supported.
          :param population: Population, membrane traces and spikes are plotted for.
          :param segment_id: Index of the neo segment to be plotted. Defaults to
                             -1, encoding the last recorded segment.
          :param ylim: y-axis limits for the plot.
          """
          if len(population) != 1:
              raise ValueError("Plotting is supported for populations of size 1.")
          # Experimental results are given in the 'neo' data format
          mem_v = population.get_data("v").segments[segment_id].irregularlysampledsignals[0]
          spikes = population.get_data("spikes").segments[-1].spiketrains[0]

          plt.plot(mem_v.times, mem_v, alpha=0.5, color='black')

          # indicate the spikes as red dots
          plt.scatter(spikes, np.max(mem_v)*np.ones_like(spikes), color='red')
          plt.xlabel("Wall clock time [ms]")
          plt.ylabel("ADC readout [a.u.]")
          if ylim:
              plt.ylim(ylim)


For our first experiment, we create a single neuron and record its spikes as well as its membrane potential.

.. code:: ipython3

   def experiment(title, v_leak, v_threshold, v_reset, i_bias_leak, savefig = False):
       """
       Set up a leak over threshold neuron.

       :param v_leak: Leak potential.
       :param v_threshold: Spike threshold potential.
       :param v_reset: Reset potential.
       :param i_bias_leak: Controls the leak conductance (membrane time constant).
       :param savefig: Save the experiment figure.
       """

       plt.figure()
       plt.title(title)

       # everything between pynn.setup() and pynn.end()
       # below is part of one hardware run.
       pynn.setup()

       # a pynn.Population corresponds to a certain number of
       # neuron circuits on the chip
       pop = pynn.Population(1, pynn.cells.HXNeuron(
           # Leak potential, range: 400-1000
           leak_v_leak=v_leak,
           # Leak conductance, range: 0-1022
           leak_i_bias=i_bias_leak,
           # Threshold potential, range: 0-500
           threshold_v_threshold=v_threshold,
           # Reset potential, range: 300-1000
           reset_v_reset=v_reset,
           # Membrane capacitance, range: 0-63
           membrane_capacitance_capacitance=63,
           # Refractory time (counter), range: 0-255
           refractory_period_refractory_time=255,
           # Enable reset on threshold crossing
           threshold_enable=True,
           # Reset conductance, range: 0-1022
           reset_i_bias=1022,
           # Increase reset conductance
           reset_enable_multiplication=True))

       pop.record(["v", "spikes"])

       # this triggers a hardware execution
       # with a duration of 0.2 ms
       pynn.run(0.2)
       plot_membrane_dynamics(pop, ylim=(100, 800))

       if savefig:
           plt.savefig(results_folder.joinpath(f'fp_task1_{time.strftime("%Y%m%d-%H%M%S")}.png'))

       plt.show()

       mem = pop.get_data("v").segments[-1].irregularlysampledsignals[0]
       spikes = pop.get_data("spikes").segments[-1].spiketrains[0]
       pynn.end()

       return mem, spikes

.. only:: jupyter

   .. code:: ipython3

       from ipywidgets import interact, IntSlider, FloatSlider
       from functools import partial
       IntSlider = partial(IntSlider, continuous_update=False)

       interact(
           experiment,
           title="Task 1: Leak over threshold",
           v_leak=IntSlider(min=400, max=1022, step=1, value=1000),
           v_threshold=IntSlider(min=0, max=500, step=1, value=500),
           v_reset=IntSlider(min=300, max=1022, step=1, value=400),
           i_bias_leak=IntSlider(min=0, max=1022, step=1, value=150),
       )


Exercises
~~~~~~~~~

A continuously spiking neuron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a first exercise, set the neuron up such that it spikes continuously without any external input.
For that purpose, play with the function's parameters and note down a suitable configuration in the following cell.
Make sure to also save the resulting plot.

.. code:: ipython3

    mem, spikes = experiment(
        title="A continuously spiking neuron",
        v_leak = 1000,
        v_threshold = 500,
        v_reset = 400,
        i_bias_leak = 150,
        savefig = True
    )

Simple spike train statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Calculate the neuron's average firing rate.
- Derive the inter-spike intervals (ISIs) of the spike train recorded in the previous exercise.
  The inter-spike interval denotes the time between two consecutive spikes of a neuron.
  Plot a histogram of the ISIs.
- Identify a method to calculate the mean and standard deviation of the neuron's instantaneous firing rate.
  The instantaneous firing rate provides a moment-by-moment measure of the neuron’s activity.

Hint: You may use, e.g., `np.diff <https://numpy.org/doc/stable/reference/generated/numpy.diff.html>`_, `np.mean <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_, and `np.std <https://numpy.org/doc/stable/reference/generated/numpy.std.html>`_ to complete these tasks.

.. only:: Solution

   **Solution:**

   .. code:: ipython3

       import numpy as np

       # calculate the interspike interval
       isi = np.diff(spikes)

       print(f"Mean firing rate: {np.mean(1/isi):.2f} +- {np.std(1/isi):.2f} kHz")

Recording the f-I curve of a silicon neuron
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we want to study the behavior of a single neuron stimulated with a current source.
For this, we introduce a new parameter for our neuron model.
This will allow to enable/disable a constant current source and observe the impact.

- Add `constant_current_enable` (True/False) and `constant_current_i_offset` (range=0-1022) as parameters into your experiment
- Configure a non firing neuron
- Vary the current and observe the impact

.. code:: ipython3

    interact(
        experiment,
        title="Task 2: Current introduced firing",
        v_leak=IntSlider(min=400, max=1022, step=1, value=1000),
        v_threshold=IntSlider(min=0, max=500, step=1, value=500),
        v_reset=IntSlider(min=300, max=1022, step=1, value=400),
        i_bias_leak=IntSlider(min=0, max=1022, step=1, value=150),
        current=IntSlider(min=0, max=1022, step=1, value=300),
    )

- To quantify your observations write a program that sweeps over a current range from 0 up to 800.
- Plot the neuron's mean instantaneous firing rate including its error over current.


.. only:: Solution

    Note: Current range limited to 800 because current source on chip looses linearity higher than that

    .. code:: ipython3

        runtime = 1

        i  = np.arange(0, 800, 50)
        f  = np.zeros_like(i, dtype=float)
        df = np.zeros_like(i, dtype=float)

        for n, i_offset in enumerate(i):

            # Required atleast on FP Setup to achieve a f-I-curve as expected
            pynn.setup(initial_config=calib)

            pop = pynn.Population(1, pynn.cells.HXNeuron(
                constant_current_enable=True,
                constant_current_i_offset=i_offset,
                refractory_period_refractory_time=50
            ))

            pop.record("spikes")

            pynn.run(runtime)
            spikes = pop.get_data().segments[-1].spiketrains[-1]
            pynn.end()

            isi = np.diff(spikes)

            if len(spikes) > 0:

                f[n] = np.mean(1/isi)
                df[n] = np.std(1/isi)

            # print(f"i:{i_offset} / f=({f[n]:.2f} +- {df[n]:.2f})kHz")

        plt.errorbar(i, f, yerr=df,
            marker='.', capsize=4, capthick=2, linestyle="None")
        plt.title("Task 2: f-I Curve")
        plt.xlabel("Current [c.u.]")
        plt.ylabel("Firing Rate [kHz]")


Synaptic stimuli and PSP stacking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the previous tasks, we analyzed the spiking dynamics of an isolated neuron.
In this task, we want to look at the membrane response of the neuron to stimulations of incoming spikes.
Your goal is to reproduce the image of PSP-Stacking (summation) from the introduction.
For this, you might find it helpful to look into pynn_introduction again.

A remark about the calibration loaded above:
While this calibration is not needed for the simple experiments above, it is essential when looking at synaptic inputs.
Therefore, you should load it when setting up this experiment using: ``pynn.setup(initial_config=calib)``.

Steps:

- Set up a population of one neuron with the default neuron parameters to record its membrane potential.
- Create a population of one `SpikeSourceArray` as a stimulating population.
- Connect the stimulating population to the neuron population.
- Find a suitable spike pattern to recreate the plot (you might require multiple spikes).
- Can the neuron spike with the current neuron configuration?
  Explain.
- Suggest and implement changes in the configuration so that the neuron spikes.
  Try to apply one change at a time.
  Explain your choices.
- Can you tell which changes can be theoretically equivalent?

*Hint*: Changes can be related to neuron parameters, projection, or population.

.. only:: jupyter

    .. code::

        # Write here your solution

.. only:: Solution

    Solution:
    ~~~~~~~~~

    **For my setup I used one target neuron and 3 spiking neurons with a continuous spiking pattern.**

    .. code::

        pynn.setup(initial_config=calib)

        neuron_parameters = {                          # range
        "leak_v_leak":200,
        "threshold_v_threshold": 400,              # (0-600)
        "threshold_enable": True,
        "refractory_period_refractory_time": 50,
        }

        neuron_type = pynn.cells.HXNeuron(**neuron_parameters)

        # save the configured neuron in the population 'pop'
        numb_neurons = 1
        pop = pynn.Population(numb_neurons, neuron_type)

        spike_times = np.arange(0, .12, .017)
        pop_stim = pynn.Population(3, pynn.cells.SpikeSourceArray(spike_times=spike_times))

        pynn.Projection(
            pop_stim,
            pop,
            pynn.AllToAllConnector(),
            synapse_type = pynn.synapses.StaticSynapse(weight=63),
            receptor_type="excitatory"
        )

        pop.record("v")

        pynn.run(.3)

        raw_data = pop.get_data()
        pynn.end()

        analog_data = raw_data.segments[-1].irregularlysampledsignals[-1]

        plt.title("PSP Stacking of Neuron")
        plt.xlabel("Wall clock time [sec]")
        plt.ylabel("Membrane Potential [a. u.]")

        plt.plot(analog_data.times, analog_data)

    **The neuron can spike with the default neuron parameters and 1 neuron in the stimulating population in the presence of sufficiently close input spikes.
    Here I try to list all suggested changes:**

    - **Decrease the time between spikes, in case this was not already implemented.**
    - **Increase the number of spikes, in case this was not already implemented.**
    - **Increase leak potential, everything else kept unchanged.**
    - **Decrease threshold potential, everything else kept unchanged.**
    - **Increase the synaptic weight, in case it was not at the maximum.
      This can lead to spiking on its own if spikes were sufficient.**
    - **Increase the number of neurons in the stimulating population.**


    **Theoretically-equivalent suggestions:**

    - **Increasing leak potential and decreasing threshold potential.**
    - **Increasing the synaptic weight and increasing the number of neurons in the stimulating population, up to a certain extent.
      The effect of the latter can be higher.
      However, this is only true if we are only concerned about spiking.**

