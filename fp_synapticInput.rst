Synaptic Input for Neurons
==========================

In this task you will investigate a single neuron receiving *excitatory* or *inhibitory* input. We begin with a situation in which the neuron does not fire in response to input, by disabling the reset mechanism and then in the last section investigate firing in response to input. The goal of the task is to familiarise yourself with the analog non-idealities that arise from a physical realisation of the idealised model equations. You will characterise different sources of noise in the system and get a feeling for the influence the hardware parameters on the temporal behaviour.

Setup
~~~~~

Generally speaking you will need to run this cell only at the beginning or after something is broken. The easiest way to escape from a broken state, will be to select "Kernel -> Restart" in the menu above and reexecute the following two cells.


.. include:: common_quiggeldy_setup.rst
.. include:: common_nightly_calibration.rst

We will also setup a folder to save experimental results in

.. code:: ipython3

    import pathlib
    
    results_folder = pathlib.Path('results/task3')
    results_folder.mkdir(parents=True, exist_ok=True)

Plotting
--------

.. code:: ipython3

    %matplotlib inline

    import matplotlib.pyplot as plt
    import numpy as np

    from pynn_brainscales.brainscales2 import Population

    plt.style.use("_static/matplotlibrc")

    def plot_membrane_dynamics(times: np.array, voltages: np.array):
        """

        :param times: sample times
        :param voltages: voltages to plot

        """
        plt.plot(times, voltages, color='grey')
        plt.xlabel("Wall clock time [ms]")
        plt.ylabel("ADC readout [a.u.]")

    def plot_population_dynamics(population: Population, segment_id=-1, ylim=None):
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

        plt.plot(mem_v.times, mem_v, alpha=0.1, color='black')

        # indicate the spikes as red dots
        plt.scatter(spikes, np.max(mem_v)*np.ones_like(spikes), color='red')
        plt.xlabel("Wall clock time [ms]")
        plt.ylabel("ADC readout [a.u.]")
        if ylim:
            plt.ylim(ylim)

Synaptic Parameters
~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import numpy as np
    from pathlib import Path
    import quantities as pq

    import pynn_brainscales.brainscales2 as pynn
    
    from pynn_brainscales.brainscales2 import Population
    from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
    from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse


Create a neuron and inject a single spike.

.. code:: ipython3

    # save results of this experiment in subfolder
    exercise_folder = results_folder.joinpath('synaptic_parameters')
    exercise_folder.mkdir(parents=True, exist_ok=True)



Experiments
-----------

Adjust the synaptic parameters and display the membrane traces.

* What do the different parameters control?

By setting the "weight" parameter to be negative the neuron will receive 
inhibitory synaptic input.

* What differences compared to the excitatory input can you observe?

Now enable the membrane threshold and adjust it in such a way that you ellicit one spike.


.. code:: ipython3

    from ipywidgets import interact, IntSlider
    from functools import partial
    IntSlider = partial(IntSlider, continuous_update=False)
    import tqdm.notebook as tqdm


    @interact(
        tau=IntSlider(min=1, max=1022, step=1, value=26),
        gm=IntSlider(min=1, max=1022, step=1, value=1022),
        weight=IntSlider(min=-63, max=63, step=1, value=63),
        n_runs=IntSlider(min=1, max=20, step=1, value=1),
        threshold_v_threshold=IntSlider(min=0, max=600, step=1, value=300),
    )
    def experiment(
        tau,
        gm,
        weight,
        n_runs,
        threshold_v_threshold,
        threshold_enable=False,
        plot=True,
        savefig=False,
        results=False
    ):
        synapse_type = 'inhibitory' if weight < 0 else 'excitatory'
        # parameters
        experiment_duration = 0.4  # ms (hw domain)
        pynn.setup(initial_config=calib)
        neuron = pynn.Population(1, pynn.cells.HXNeuron(
            excitatory_input_i_bias_tau=tau,
            excitatory_input_i_bias_gm=gm,
            inhibitory_input_i_bias_tau=tau,
            inhibitory_input_i_bias_gm=gm,
            # neuron to be observed: disable threshold such we can observe PSP shape
            threshold_enable=threshold_enable,
            threshold_v_threshold=threshold_v_threshold
        ))
        neuron.record(["v", "spikes"])

        # external input
        spike_times = [experiment_duration / 2]
        pop_input = pynn.Population(1,
            pynn.cells.SpikeSourceArray(spike_times=spike_times))

        synapse = pynn.standardmodels.synapses.StaticSynapse(weight=weight)
        proj = pynn.Projection(pop_input, neuron, pynn.AllToAllConnector(),
                           synapse_type=synapse, receptor_type=synapse_type)

        # run experiment
        voltages = []
        mems = []
        min_shape = 2**32

        for run in tqdm.trange(n_runs, leave=False):
            pynn.run(experiment_duration)
            mem = neuron.get_data("v").segments[-1].irregularlysampledsignals[0]
            mems.append(mem)
            voltages.append(np.array(mem))
            min_shape = min(mem.shape[0], min_shape)
            pynn.reset()

        voltages = [v[:min_shape] for v in voltages]
        voltages = np.stack(voltages)

        if plot:
            plot_membrane_dynamics(mems[-1].times[:min_shape], np.mean(voltages, axis=0))

        if savefig:
            fig.savefig(exercise_folder.joinpath('membrane_trace.png'))

        pynn.end()
        if results:
            return mems, voltages, mems[-1].times[:min_shape]

Solution 
--------

Please enter your answers to the questions above here...


Normalising the Membrane Dynamics 
---------------------------------
We can use the experiment above to determine the minimum and maximum of the membrane dynamics over a number of runs.

.. code:: ipython3

    def determine_vmax_vmin(tau, gm, weight, v_threshold, n_runs):
        mems, voltages, _ = experiment(tau=tau, gm=gm, weight=weight, n_runs=n_runs,    threshold_enable=True, threshold_v_threshold=v_threshold, results=True, plot=False)
        v_max, v_min = np.max(voltages[:,:,0]), np.min(voltages[:,:,0])
        return v_min, v_max

    v_min, v_max = determine_vmax_vmin(1, 1022, weight=63, v_threshold=300, n_runs=10)

This can be used to normalise the membrane voltage measurements. Note that this normalisation depends on the choices of hyperparameters $\\tau$, $g_m$, $v_\\mathrm{th}$ used to obtain it. The idea is to map $v_\\mathrm{th}$ to $1$ and $v_\\mathrm{reset}$ to $0$, although this will only be approximately true for all membrane traces.

.. code:: ipython3

    def normalise_voltage(v_min, v_max):
        def normalise(v):
            delta_v = v_max - v_min
            return 1/delta_v * (v - v_min)

        return normalise

    nv = normalise_voltage(v_min, v_max)


Trial-to-Trial Variations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    taus = np.arange(1,1022,100)
    gm = 1022
    weight = 63
    n_runs = 5
    max_voltages = []
    baselines = []

    for tau in tqdm.tqdm(taus):
        m, v, _ = experiment(tau=tau, gm=gm, weight=weight, n_runs=n_runs, threshold_v_threshold=300, plot=False, results=True)
        v_baseline = v[:100].mean()
        mean = np.mean(v, axis=0)
        max_v = np.max(mean, axis=0)
        max_voltages.append(max_v[0])
        baselines.append(v_baseline)


.. code:: ipython3

    plt.plot(taus, np.array(max_voltages) - np.array(baselines))


.. code:: ipython3


    tau = 26
    gm = 1022
    weight = 63
    n_runs = 200 # each run takes some time (with this value it should finish in less than 1-2 minutes)

    m, v, _ = experiment(tau=tau, gm=gm, weight=weight, n_runs=n_runs, threshold_v_threshold=300, plot=False, results=True)
    v_baseline = v[:100].mean()
    max_v = np.max(v-v_baseline, axis=1)

    def histogram(heights):
        fig, ax = plt.subplots()
        ax.set_xlabel("height of peaks [MADC]")
        ax.set_ylabel("counts")

        ax.hist(heights)

        return fig

    fig = histogram(max_v)

We can also sweep weight dependence of synaptic input

.. code:: ipython3

    tau = 26
    gm = 1022
    n_runs = 10
    means = []
    n_sub = 20
    ws = np.linspace(-64,64,n_sub)
    category_colors = plt.get_cmap('coolwarm')(0.5+ws/128)

    max_mean_v = []

    for idx in tqdm.trange(n_sub):
        m, v, times = experiment(tau=tau, gm=gm, weight=ws[idx], n_runs=n_runs, threshold_v_threshold=300, plot=False, results=True)
        v_baseline = v[:100].mean()

        for j in range(n_runs):
            plt.plot(times, v[j] - v_baseline, color=category_colors[idx], alpha=.1)

        plt.plot(times, np.mean(v, axis=0) - v_baseline, color=category_colors[idx])

        sign = -1 if ws[idx] < 0 else 1

        max_mean_v.append(sign*np.max(np.abs(np.mean(v, axis=0) - v_baseline)))

    plt.xlabel("Wall clock time [ms]")
    plt.ylabel("ADC readout [a.u.]")

Exercises
---------
* Plot the maximum of the voltages we obtained

Solution
--------
Please enter your solution here...

.. only:: Solution

    .. code:: ipython3

        plt.plot(ws, max_mean_v)
        plt.ylabel("ADC readout [a.u.]")
        plt.xlabel("Weight")

.. only:: not Solution
    
    .. code:: ipython3

        ... # TODO



Fixed-Pattern Noise
~~~~~~~~~~~~~~~~~~~

Investigate the fixed pattern noise between synapses/synapse drivers.

.. code:: ipython3

    def extract_heights(voltage, spike_times):
        '''
        Extract the PSP heights from a membrane trace where several inputs
        where injected one after another.

        :param voltage: Recorded membrane trace
        :param spike_times: Input spike times for the different inputs in ms
            (hw domain).
        '''

        # determine baseline voltage
        v_baseline = voltage[:100].mean()

        # extract for each synapse the maximum in the membrane trace and
        # substact the baseline voltage
        heights = []

        # determine how many samples are recorded between inputs
        idx_first_input = np.argmin(np.abs(voltage.times - spike_times[0] * pq.ms))
        samples_per_input = 2 * idx_first_input

        # loop over inputs
        for n_input in range(len(spike_times)):
            start_stop = (np.array([0, 1]) + n_input) * samples_per_input
            input_slice = slice(start_stop[0], start_stop[1])
            height = np.max(voltage[input_slice]) - v_baseline
            heights.append(float(height))

        return heights


.. code:: ipython3

    # save results of this experiment in subfolder
    exercise_folder = results_folder.joinpath('fixed_pattern_noise')
    exercise_folder.mkdir(parents=True, exist_ok=True)

    def fixed_pattern_noise_experiment():
        # parameters
        n_synapses = 100
        weight = 63
        time_between_inputs = 0.4

        pynn.setup(initial_config=calib)

        # neuron to be observed: disable threshold such we can observe PSP shape
        neuron = pynn.Population(1, pynn.cells.HXNeuron(threshold_enable=False))
        neuron.record("v")

        # external input
        # since we want to test several external synapses, we create a population
        # of external neurons which spike one after another
        spike_times = (np.arange(n_synapses) + 0.5) * time_between_inputs
        # SpikeSourceArray expects a list of spike times for each neuron in the
        # population -> reshape
        pop_input = pynn.Population(n_synapses,
            pynn.cells.SpikeSourceArray(spike_times=spike_times.reshape([-1, 1]).tolist()))

        synapse = pynn.standardmodels.synapses.StaticSynapse(weight=weight)
        proj = pynn.Projection(pop_input, neuron, pynn.AllToAllConnector(),
                               synapse_type=synapse, receptor_type='excitatory')

        # run experiment
        pynn.run(n_synapses * time_between_inputs)

        # save membrane trace
        mem = neuron.get_data("v").segments[-1].irregularlysampledsignals[0]
        np.savetxt(exercise_folder.joinpath('membrane_trace.txt'), mem.base)
        pynn.end()

        return mem, spike_times

Exercises
---------

Extract the PSP height from the trace and plot it in a histogram.
You can use the functions you implemented above.

Hints:
* The function to plot histograms with matplotlib is called `hist <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html>`_

Solution
--------

Please enter your solution here... 

.. only:: Solution

    .. code:: ipython3

        mem, spike_times = fixed_pattern_noise_experiment()
        heights = extract_heights(mem, spike_times)
        plt.hist(heights)
        plt.savefig(exercise_folder.joinpath('histogram.png'))

.. only:: not Solution
    
    .. code:: ipython3
        
        mem, spike_times = fixed_pattern_noise_experiment()
        heights = ...
        # TODO plot and save histogram

Stacking of PSPs
~~~~~~~~~~~~~~~~


.. code:: ipython3

    @interact(exc_weight=IntSlider(min=0, max=63, step=1, value=31),
              inh_weight=IntSlider(min=0, max=63, step=1, value=31),
              tau=IntSlider(min=1, max=1022, step=1, value=26),
              gm=IntSlider(min=1, max=1022, step=1, value=1022),
              isi=IntSlider(min=10, max=100, step=5, value=50),
              n_runs=IntSlider(min=1, max=10, step=1, value=1)
              )
    def run_experiment(exc_weight: int, inh_weight: int, tau: float, gm: float, isi: float, n_runs: int, plot = True, return_results = False):
        '''
        Run external input demonstration on BSS-2.

        Adjust weight of projections, set input spikes and execute experiment
        on BSS-2.

        :param exc_weight: Weight of excitatory synaptic inputs, value range
            [0,63].
        :param inh_weight: Weight of inhibitory synaptic inputs, value range
            [0,63].
        :param isi: Time between synaptic inputs in microsecond (hardware
            domain)
        '''

        plt.figure()
        plt.title("Fourth experiment: External stimulation")

        pynn.setup(initial_config=calib)

        # use calibrated parameters for neuron
        stimulated_p = pynn.Population(1, pynn.cells.HXNeuron(
            excitatory_input_i_bias_tau=tau,
            excitatory_input_i_bias_gm=gm,
            inhibitory_input_i_bias_tau=tau,
            inhibitory_input_i_bias_gm=gm
        ))
        stimulated_p.record(["v", "spikes"])

        # calculate spike times
        wait_before_experiment = 0.01  # ms (hw)
        isi_ms = isi / 1000  # convert to ms
        spiketimes = np.arange(6) * isi_ms + wait_before_experiment

        # all but one input are chosen to be exciatory
        excitatory_spike = np.ones_like(spiketimes, dtype=bool)
        excitatory_spike[1] = False

        # external input
        exc_spikes = spiketimes[excitatory_spike]
        exc_stim_pop = pynn.Population(1, SpikeSourceArray(spike_times=exc_spikes))
        exc_proj = pynn.Projection(exc_stim_pop, stimulated_p,
                                   pynn.AllToAllConnector(),
                                   synapse_type=StaticSynapse(weight=exc_weight),
                                   receptor_type="excitatory")

        inh_spikes = spiketimes[~excitatory_spike]
        inh_stim_pop = pynn.Population(1, SpikeSourceArray(spike_times=inh_spikes))
        inh_proj = pynn.Projection(inh_stim_pop, stimulated_p,
                                   pynn.AllToAllConnector(),
                                   synapse_type=StaticSynapse(weight=-inh_weight),
                                   receptor_type="inhibitory")

        # run experiment

        # run experiment
        voltages = []
        mems = []
        min_shape = 2**32
        experiment_duration = 0.6

        for run in range(n_runs):
            pynn.run(experiment_duration)
            mem = stimulated_p.get_data("v").segments[-1].irregularlysampledsignals[0]
            if plot:
                plot_population_dynamics(stimulated_p, ylim=(100, 600))
            mems.append(mem)
            voltages.append(np.array(mem))
            min_shape = min(mem.shape[0], min_shape)
            pynn.reset()

        voltages = [v[:min_shape] for v in voltages]
        voltages = np.stack(voltages)

        if return_results:
            return voltages



Exercises
---------

* Adjust the time between the synaptic inputs and investigate when the neuron is firing.
* Save a plot in which you can observe firing behaviour and save the parameters.

Solution
--------

Document your observations here...


Behaviour under Stimulation by a Poisson Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So far we have only looked at a single neuron receiving deterministic input. In this last exercise for this section of the lab course, we will look at one neuron receiving "stochastic"
input from two sources: an inhibitory and excitatory input source emitting spikes according to a poisson process with intensity $\\lambda_e,\\lambda_i$. Poisson processes occur frequently
in nature and have wide `applications in many areas <https://en.wikipedia.org/wiki/Poisson_point_process>`_. One key point of this exercise
therefore is to give you an idea how a spiking neuron running on Neuromorphic Hardware 
could naturally and efficiently process event data.

.. code:: ipython3

    def poisson_spike_times(lambda0 = 100, delta_t = 0.6):
        n_spikes = np.random.poisson(lambda0 * delta_t)
        times = delta_t * np.random.uniform(0, 1, n_spikes)
        return np.sort(times)
    
The following code defines the experiment:

.. code:: ipython3

    @interact(exc_weight=IntSlider(min=0, max=63, step=1, value=31),
          inh_weight=IntSlider(min=0, max=63, step=1, value=31),
          lambda_exc=IntSlider(min=1, max=200, step=1, value=50),
          lambda_inh=IntSlider(min=1, max=200, step=1, value=50),
          n_runs=IntSlider(min=1, max=10, step=1, value=1)
          )
    def run_experiment(exc_weight: int, inh_weight: int, lambda_exc: float, lambda_inh:float, n_runs: int, plot = True, return_results = False):
        '''
        Run external input demonstration on BSS-2.

        Adjust weight of projections, set input spikes and execute experiment
        on BSS-2.

        :param exc_weight: Weight of excitatory synaptic inputs, value range
            [0,63].
        :param inh_weight: Weight of inhibitory synaptic inputs, value range
            [0,63].
        :param lambda_exc: excitatory poisson process intensity
        :param lambda_inh: inhibitory poisson process intensity
        '''

        plt.figure()
        plt.title("Fourth experiment: External stimulation")

        pynn.setup(initial_config=calib)

        # use calibrated parameters for neuron
        stimulated_p = pynn.Population(1, pynn.cells.HXNeuron())
        stimulated_p.record(["v", "spikes"])

        # calculate spike times
        experiment_duration = 0.6
        wait_before_experiment = 0.01  # ms (hw)

        # external input
        exc_stim_pop = pynn.Population(1, SpikeSourceArray(spike_times=[]))
        exc_proj = pynn.Projection(exc_stim_pop, stimulated_p,
                                   pynn.AllToAllConnector(),
                                   synapse_type=StaticSynapse(weight=exc_weight),
                                   receptor_type="excitatory")

        inh_spike_source = SpikeSourceArray(spike_times=[])
        inh_stim_pop = pynn.Population(1, inh_spike_source)
        inh_proj = pynn.Projection(inh_stim_pop, stimulated_p,
                                   pynn.AllToAllConnector(),
                                   synapse_type=StaticSynapse(weight=-inh_weight),
                                   receptor_type="inhibitory")

        # run experiment

        # run experiment
        voltages = []
        mems = []
        min_shape = 2**32


        for run in tqdm.trange(n_runs):  
            inh_spikes = poisson_spike_times(lambda_inh, experiment_duration) + wait_before_experiment
            inh_stim_pop.set(spike_times = inh_spikes)
            exc_spikes = poisson_spike_times(lambda_exc, experiment_duration) + wait_before_experiment
            exc_stim_pop.set(spike_times = exc_spikes)

            pynn.run(experiment_duration)
            mem = stimulated_p.get_data("v").segments[-1].irregularlysampledsignals[0]
            if plot:
                plot_population_dynamics(stimulated_p, ylim=(100, 600))
            mems.append(mem)
            voltages.append(np.array(mem))
            min_shape = min(mem.shape[0], min_shape)
            pynn.reset()

        voltages = [v[:min_shape] for v in voltages]
        voltages = np.stack(voltages)

        if return_results:
            return voltages

Exercises
---------
Explore the parameters offered by the experiment and describe your observations.
- Consider multiple runs (n > 1), how would you characterise the behaviour over multiple runs.
- What happens if the intensity of the inhibitory and excitatory input differ widely?
- What happens if you change the weight of the inhibitory / excitatory input?
- What quantitative ways of analysing the observed behaviour come to your mind?

Solutions
---------
Please give your solutions here...

Analysis of Stimulation by a Poisson Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following gives you a way to record membrane voltage traces 
for given paramters over many runs (n_runs = 1000)

.. code:: ipython3

    voltages = run_experiment(exc_weight=31, inh_weight=31, lambda_exc=50, lambda_inh=50, n_runs=1000, return_results=1)

Exercises
---------

* Plot a histogram of the voltages, for different parameter settings. Consider both population (over all runs) and a histograms over time for a single run. 
* How could you analyse the observed distribution further?

Solution 
--------

Please type your solution here...

Hint: Consider plotting the histogram with density=True).

.. only:: Solution

    .. code:: ipython3

        plt.hist(voltages[:,:,0].flatten(), bins=100, density=True)

.. only:: not Solution

    .. code:: ipython3

        # TODO
        plt.hist(...)
