Calibration: Characterising and Measuring Analog Behaviour
==========================================================

The goal of the exercises assembled in this task is to familiarise yourself with how we characterise and measure the analog behaviour of the chip. At the end you will understand
how to convert raw hardware measurements into physical units and how to relate digital calibration values to measured membrane traces.


.. include:: common_note_helpers.rst

.. only:: jupyter

   Setup
   ~~~~~
   .. include:: common_quiggeldy_setup.rst


Characterizing the MADC
~~~~~~~~~~~~~~~~~~~~~~~

Before we continue our investigations of analog neurons, we need to characterize the readout chain used for the neurons' signals. Since HICANN-X has digital communication interfaces, the analog voltage recordings of a neuron need to be digitized on chip.

Characterizing the analog-to-digital-converter (ADC) means applying a known voltage externally and recording the value returned by the ADC. Later, we will reverse this relation in order to convert measured data back to SI units.

In the following, we define a function which applies an external voltage and returns samples from the MADC. Your task is to sweep the external voltage and analyze the returned reads. Plot the characteristic and perform a linear fit to a suitable range. You will use the parameters of your fit in the following notebooks.

Low Level Hardware Access
^^^^^^^^^^^^^^^^^^^^^^^^^

You will not need to make changes to the function imported in the next cell. If you are curious how it is implemented you can take a `look at it <_static/common/measure_voltage.py>`_.


.. code:: ipython3

   from _static.common.measure_voltage import measure_voltage

We first test the function above by setting a medium voltage, say 0.5 V. Look at the printed samples and their data type - notice the sampled value is the first entry in each tuple, accessible by the named field "value".

.. code:: ipython3

   
   from dlens_vx_v3 import hxcomm

   # since we're not using the PyNN frontend here, we need to define our own
   # connection to the hardware. It is available as a context manager:
   with hxcomm.ManagedConnection() as connection:
       result = measure_voltage(connection, 0.5)
   print(result[:20])
   print(result.dtype)

Exercises
^^^^^^^^^

* Measure the sampled value for voltages between 0 and 1.2 V. We recommend steps of 0.05 V. For each data point, save a mean ADC value. Hint: Use something like ``result["value"]`` for calculating your mean ADC values. Complete the cell below to do so:

Hint: You can use the measure_voltage function.

.. only:: not Solution

   .. code:: ipython3

      import numpy as np

      # Measure characteristic
      voltages =  ... # volt
      results = np.zeros_like(voltages)  # mean ADC values in LSB
      errors = np.zeros_like(voltages)  # std deviation of ADC values in LSB

      with hxcomm.ManagedConnection() as connection:
            for voltage_id, voltage in enumerate(voltages):
              samples = ... 
              results[voltage_id] = ...
              errors[voltage_id] = ...
          ...

* Plot the characteristic, i.e. plot the acquired value over the external voltage.

.. only:: not Solution

    .. code:: ipython3

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        
        # TODO
        ...

        plt.xlabel("Voltage [V]")
        plt.ylabel("MADC value [LSB]")

* Perform a linear fit in a suitable range (exclude possibly saturated values on top and bottom). Hint: In order to convert sample values to voltage later, you can use the results as "x data" and the voltages as "y data" for your fit function.

.. only:: not Solution

    .. code:: ipython3

      # fit linear function to data
      from scipy.optimize import curve_fit

      # TODO

* Plot your linear fit and save the plot.
* Save the fit parameters for later, so you can convert ADC values to Volt.


.. only:: Solution

   .. code:: ipython3

      import numpy as np


      # Measure characteristic
      voltages = np.arange(0, 1.21, 0.05)  # volt
      results = np.zeros_like(voltages)  # mean ADC values in LSB
      errors = np.zeros_like(voltages)  # std deviation of ADC values in LSB

      with hxcomm.ManagedConnection() as connection:
          for voltage_id, voltage in enumerate(voltages):
              samples = measure_voltage(connection, voltage)
              results[voltage_id] = np.mean(samples["value"])
              errors[voltage_id] = np.std(samples["value"])

      # plot data
      import matplotlib.pyplot as plt
      plt.figure(figsize=(12, 8))
      plt.errorbar(voltages, results, yerr=errors, fmt='.', label="data")
      plt.xlabel("Voltage [V]")
      plt.ylabel("MADC value [LSB]")

      # fit linear function to data
      from scipy.optimize import curve_fit
      linear = lambda x, m, b: m * x + b

      popt, _ = curve_fit(linear, results[5:-5], voltages[5:-5])

      # plot fit
      yvalues = np.array([np.min(results), np.max(results)])
      fit_label = f"V = {popt[0]:.5f} * value " + \
        ("+" if popt[1] > 0 else "-") + f" {np.abs(popt[1]):.3f}"
      plt.plot(linear(yvalues, *popt), yvalues, label=fit_label)
      _ = plt.legend()

You can use the fit parameters you obtained to implement a madc value to voltage conversion routine like so:

.. code:: ipython3

    def madc_to_voltage_conversion(m, b):
        def convert(value):
            return m * value + b
        return convert

    madc_to_v = madc_to_voltage_conversion(*popt)


Characterizing Neuron Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We now move on to a characterisation of Neuron Parameters, the goal is to relate the 
parameter values, which are specified in arbitrary units to the MADC measurement values.
That way you will establish a correspondence between the values you specify and the membrane 
measurement values send back to you by the chip.

We begin by importing some commonly used calibration code

.. include:: common_nightly_calibration.rst


and define plotting functions

.. code:: ipython3

    import pynn_brainscales.brainscales2 as pynn
    
    from pynn_brainscales.brainscales2 import Population
    from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
    from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse

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


The following experiment code let's you interactively explore the measurement we are about to perform:

.. code:: ipython3

    from ipywidgets import interact, IntSlider
    from functools import partial
    IntSlider = partial(IntSlider, continuous_update=False)

    @interact(
        neuron_idx=IntSlider(min=0, max=511, step=1, value=0),
        v_leak=IntSlider(min=0, max=1022, step=1, value=1000),
        v_threshold=IntSlider(min=0, max=500, step=1, value=500),
        v_reset=IntSlider(min=0, max=1022, step=1, value=400),
        i_bias_leak=IntSlider(min=0, max=1022, step=1, value=150),
    )
    def experiment(neuron_idx, v_leak, v_threshold, v_reset, i_bias_leak):
        """
        Set up a leak over threshold neuron.

        :param v_leak: Leak potential.
        :param v_threshold: Spike threshold potential.
        :param v_reset: Reset potential.
        :param i_bias_leak: Controls the leak conductance (membrane time constant).
        :param savefig: Save the experiment figure.
        """

        plt.figure()

        # everything between pynn.setup() and pynn.end() 
        # below is part of one hardware run.
        pynn.setup(initial_config=calib, neuronPermutation=[neuron_idx])

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
        plt.show()

        mem = pop.get_data("v").segments[-1].irregularlysampledsignals[0]
        spikes = pop.get_data("spikes").segments[-1].spiketrains[0]
        pynn.end()

        return mem, spikes   


We now define a function which returns the measured membrane voltage trace for a given set of parameters

.. code:: ipython3

    def get_v_mem(neuron_params: dict, neuron_idx: int) -> np.ndarray:
        pynn.setup(initial_config=calib, neuronPermutation=[neuron_idx])
        pop = pynn.Population(1, pynn.cells.HXNeuron(**neuron_params))
        pop.record(["v"])
        pynn.run(0.2)
        mem_v = pop.get_data("v").segments[0].irregularlysampledsignals[0]
        pynn.end()
        return mem_v.base

We want to perform this experiment over potentially a larger number of neurons
and for specific neuron ranges

.. code:: ipython3

    def get_data(number_of_neurons, sweep_range, sweep_neuron_param,
                 filter_function, neuron_params):

        y_measured = np.zeros(shape=(number_of_neurons, len(sweep_range)))
        for neuron_idx in range(number_of_neurons):
            y_measured_local = []
            for dac_idx, dac_value in enumerate(sweep_range):
                neuron_params.update({sweep_neuron_param: dac_value})
                membrane = get_v_mem(neuron_params, neuron_idx)
                print(f"Measuring neuron {neuron_idx} for "
                      f"{sweep_neuron_param} with value: {dac_value}")
                y_measured_local.append(filter_function(membrane.magnitude))
                y_measured[neuron_idx][dac_idx] = filter_function(membrane.magnitude)

        return y_measured

This defines the main function for the parameter sweep

.. code:: ipython3

    def main(number_of_neurons, sweep_range, param_to_be_determined):
        if param_to_be_determined == "threshold":
            neuron_params = {"leak_v_leak": 1022,
                             "reset_v_reset": 50}
            sweep_neuron_param = "threshold_v_threshold"
            filter_function = np.max

        if param_to_be_determined == "reset":
            neuron_params = {"threshold_v_threshold": 150,
                             "leak_v_leak": 1022}
            sweep_neuron_param = "reset_v_reset"
            filter_function = np.min

        if param_to_be_determined == "leak":
            neuron_params = {"threshold_v_threshold": 1022,
                             "reset_v_reset": 50}
            sweep_neuron_param = "leak_v_leak"
            filter_function = np.mean

        y_measured = get_data(number_of_neurons, sweep_range, sweep_neuron_param,
                 filter_function, neuron_params)

        # 
        measured_mean = np.mean(y_measured, axis=0)
        measured_err = np.std(y_measured, axis=0)
        return sweep_range, y_measured, measured_mean, measured_err

    def plot_and_fit(ax, data, label):
        sweep_range, y_measured, measured_mean, measured_err = data

        number_of_neurons = y_measured.shape[0]

        ax.set_xlabel(f"set $V_{{{label}}}$ [DAC value]")
        ax.set_ylabel(f"measured $V_{{{label}}}$ [ADC value]")
        for k in range(number_of_neurons):
            ax.plot(sweep_range, y_measured[k], '-',
                     color='darkgray', linewidth=0.25, zorder=-1)

        ax.errorbar(sweep_range, measured_mean, yerr=measured_err,
                     color='red', linestyle='-', marker='None', zorder=0)

        linear_fit = np.polyfit(sweep_range, measured_mean, 1)
        ax.plot(sweep_range, linear_fit[0] * sweep_range + linear_fit[1],
                 'b--', zorder=1)

        return linear_fit



The general idea is to modify the sweep range to find a suitable range for the fit
for respective parameters. Start with full range and narrow down.

.. code:: ipython3

    # for how many neurons should an average characterisation be done
    num_neurons = 1

    # valid range [0, 1022]
    adc_sweep_range = np.arange(0, 1022, 50)

    # parameter to be determined, valid "threshold", "reset" or "leak"
    adc_sweep_range = np.arange(50, 301, 50)
    param_name = "threshold"
    threshold_data = main(num_neurons, adc_sweep_range, param_name)
    
    adc_sweep_range = np.arange(500, 901, 100)
    param_name = "leak"
    leak_data = main(num_neurons, adc_sweep_range, param_name)
    
    adc_sweep_range = np.arange(400, 901, 100)
    param_name = "reset"
    reset_data = main(num_neurons, adc_sweep_range, param_name)


Excercises
^^^^^^^^^^

.. code:: ipython3

    fig, ax = plt.subplots(1,3, sharey=True)

    v_th_fit = plot_and_fit(ax[0], threshold_data, "th")
    v_leak_fit = plot_and_fit(ax[1], leak_data, "leak")
    v_reset_fit = plot_and_fit(ax[2], reset_data, "reset")

.. code:: ipython3

    def linear(param):
        def fun(x):
            return param[0] * x + param[1]

        def inverse(x):
            return (x - param[1]) / param[0]

        return fun, inverse

    v_th_s2m, v_th_m2s = linear(v_th_fit)
    v_leak_s2m, v_leak_m2s = linear(v_leak_fit)
    v_reset_s2m, v_reset_m2s = linear(v_reset_fit)



.. code:: ipython3

    @interact(
        neuron_idx=IntSlider(min=0, max=511, step=1, value=0),
        v_leak=IntSlider(min=v_leak_s2m(0), max=v_leak_s2m(1022), step=1, value=v_leak_s2m(1000)),
        v_threshold=IntSlider(min=v_leak_s2m(0), max=v_th_s2m(500), step=1, value=v_th_s2m(500)),
        v_reset=IntSlider(min=v_reset_s2m(0), max=v_reset_s2m(1022), step=1, value=v_reset_s2m(400)),
        i_bias_leak=IntSlider(min=0, max=1022, step=1, value=150),
    )
    def experiment(neuron_idx, v_leak, v_threshold, v_reset, i_bias_leak):
        """
        Set up a leak over threshold neuron.

        :param v_leak: Leak potential.
        :param v_threshold: Spike threshold potential.
        :param v_reset: Reset potential.
        :param i_bias_leak: Controls the leak conductance (membrane time constant).
        :param savefig: Save the experiment figure.
        """

        v_leak = v_leak_m2s(v_leak)
        v_threshold = v_th_m2s(v_threshold)
        v_reset = v_reset_m2s(v_reset)

        plt.figure()

        # everything between pynn.setup() and pynn.end()
        # below is part of one hardware run.
        pynn.setup(initial_config=calib, neuronPermutation=[neuron_idx])

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
        plt.axhline(v_th_s2m(v_threshold))
        plt.axhline(v_reset_s2m(v_reset))
        plt.axhline(v_leak_s2m(v_leak), color='green')

        plt.show()

        mem = pop.get_data("v").segments[-1].irregularlysampledsignals[0]
        spikes = pop.get_data("spikes").segments[-1].spiketrains[0]
        pynn.end()

        return mem, spikes


Calibrating Neuron Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    def membrane2dac(voltage, slope, offset):
        return int((voltage - offset) / slope)


    # set voltages
    # threshold must be according to:
    v_leak = 350
    v_reset = 200
    v_th = v_leak - (v_leak - v_reset) / np.exp(1)

    # enter your fit results from task a) into the membrane2dac calls
    neuron_params = {"leak_i_bias": 400,
                     "reset_i_bias": 1022,
                     "leak_enable_division": True,
                     "threshold_enable": True,
                     "reset_enable_multiplication": True,
                     "membrane_capacitance_capacitance": 63,
                     "refractory_period_refractory_time": 150,
                     "leak_v_leak": membrane2dac(v_leak, 0.79, -180),
                     "reset_v_reset": membrane2dac(v_reset, 0.88, -250),
                     "threshold_v_threshold": membrane2dac(v_th, 0.99, 60)
                     }

    # the leak conductance g_leak is set by I_bias
    # each entry refers to a different neuron
    # must be adjusted for identical firing rates!
    I_bias = [400, 400, 400, 400]
    assert len(I_bias) > 3, "You need to look at least on 4 neurons."

Gather spikes an membrane trace for different neurons


.. code:: ipython3

    # as we can only readout the membrane of one neuron at a time we need to
    # repeat the measurement for each neuron

    result = []
    for nrnidx, bias in enumerate(I_bias):
        pynn.setup()

        pop = pynn.Population(len(I_bias),
                              pynn.cells.HXNeuron(**neuron_params))
        pop.set(leak_i_bias=bias)

        # a population view is a subset of a population
        p_view = pynn.PopulationView(pop, [nrnidx])
        p_view.record(["v", "spikes"])

        pynn.run(2)
        spikes = p_view.get_data("spikes").segments[0].spiketrains[0]

        # # FIXME spike readout fails some times
        # if len(spikes) <= 1:
        #     raise RuntimeError("not enough spikes recorded")

        mem_v = p_view.get_data("v").segments[0].irregularlysampledsignals[0]

        pynn.end()

        result.append((spikes, mem_v))

Analyse and plot neuron output

.. code:: ipython3

    membrane = []
    spikes = []
    isi = []

    for nrnidx, (spike, v) in enumerate(result):
        spikes.append(spike)
        membrane.append(v)
        isi.append(np.diff(spikes[nrnidx]))

    num_plots = len(I_bias)
    # -> play around with figsize to modify plot resolution
    _, axes = plt.subplots(
        num_plots, 1, sharex=True, sharey=True, figsize=(8, 5))
    axes = axes.flatten()

    for nrnidx in range(num_plots):
        max_v = np.max(membrane[nrnidx])
        print(f'neuron {nrnidx}')
        print(f"max. membrane potential: {max_v:.3f} V")
        print(f"firing rate: {np.mean(1 / isi[nrnidx]):.4} "
                f"+- {np.std(1 / isi[nrnidx]):.2f} MHz")

        axes[nrnidx].plot(membrane[nrnidx].times, membrane[nrnidx])
        axes[nrnidx].plot(
            spikes[nrnidx],
            [max_v for i in range(len(spikes[nrnidx]))],
            '.', color='red')

    axes[0].set_xlim(0, 0.2)
    axes[2].set_ylabel("V [V]", fontsize=12)
    axes[-1].set_xlabel("t [ms]", fontsize=12)

    figname = f'fp_task2b_4membranes_{time.strftime("%Y%m%d-%H%M%S")}.png'
    plt.savefig(figname)
    print(f"plot saved {figname}")


Calibrating Membrane Time Constant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this optional task you create a calibration routine to calibrate the leak potential of all neurons.



    .. code:: ipython3

        def membrane2dac(voltage, slope, offset):
            return int((voltage - offset) / slope)


        # set voltages in volt
        # threshold must be according to:
        v_leak = 350
        v_reset = 200
        v_th = v_leak - (v_leak - v_reset) / np.exp(1)

        # enter your fit results from task a) into the voltage2dac calls
        neuron_params = {"leak_i_bias": 400,
                         "reset_i_bias": 38,
                         "leak_enable_division": True,  # set False for freq > 40
                         "threshold_enable": True,
                         "reset_enable_multiplication": True,
                         "membrane_capacitance_capacitance": 63,
                         "refractory_period_refractory_time": 150,
                         "leak_v_leak": membrane2dac(v_leak, 0.79, -180),
                         "reset_v_reset": membrane2dac(v_reset, 0.88, -250),
                         "threshold_v_threshold": membrane2dac(v_th, 0.99, 60)
                        }

        targetrate = 30.0  # in kHz

        # strength of calibration
        # choose value between 0 and 1
        calib_param = 0.7

    .. code:: ipython3

        def get_data(i_bias, iteration):
            rates = []
            gleaks = []

            # capacitance from DAC value (63 equals 2.2pF)
            c_m = 2.2 * 10e-12 * neuron_params["membrane_capacitance_capacitance"] / 63

            # tau_ref from DAC value with f_clock = 10MHz
            tau_ref = neuron_params["refractory_period_refractory_time"] / (10e6) * 1e3

            pynn.setup()

            pop = pynn.Population(len(i_bias), pynn.cells.HXNeuron(**neuron_params))
            pop.set(leak_i_bias=i_bias)
            pop.record(["spikes"])

            pynn.run(2)
            all_spikes = pop.get_data("spikes").segments[0].spiketrains

            pynn.end()

            for nrnidx, spikes in enumerate(all_spikes):
                # if not enough spikes are recorded, the neuron is skipped
                if len(spikes) <= 1:
                    rates.append(-1)
                    gleaks.append(-1)

                    print("not enough spikes recorded for neuron ", nrnidx)
                else:
                    # calculate the firing rate from the spike array
                    isi = np.array(np.diff(spikes))
                    rates.append(1 / np.mean(isi))

                    # calculate gleak from the interspike interval
                    tau_m = np.mean(isi) - tau_ref
                    gleaks.append(c_m / tau_m)

                    # print(f"data obtained for neuron {nrnidx}")

            print(f"The average firing rate is {np.mean(rates):.4} "
                  f"+- {np.std(rates):.3} kHz")
            print(f"The average leak conductance is {np.mean(gleaks):.4} "
                  f"+- {np.std(gleaks):.3} S")

            # save results for plotting
            np.savetxt(f"fp_task2c_rates_iteration{iteration}.txt", rates)
            np.savetxt(f"fp_task2c_gleaks_iteration{iteration}.txt", gleaks)

    .. code:: ipython3

        def calibrate_neurons(i_bias, num_iterations):

            for iteration in range(num_iterations):
                get_data(i_bias, iteration)
                rates = np.loadtxt(f"fp_task2c_rates_iteration{iteration}.txt")
                for nrnidx, _ in enumerate(i_bias):
                    if rates[nrnidx] != -1:
                        # TODO: Check code here
                        scaling = (calib_param * (targetrate / rates[nrnidx] - 1) + 1)
                        i_bias[nrnidx] = int(scaling * i_bias[nrnidx])
                        if i_bias[nrnidx] > 1022:
                            print(f"Reached max i_bias for neuron {nrnidx}")
                            i_bias[nrnidx] = 1022
                        if i_bias[nrnidx] < 0:
                            print(f"Reached min i_bias for neuron {nrnidx}")
                            i_bias[nrnidx] = 0


    Definition of visualization

    .. code:: ipython3

        def plot_histogram(num_iterations):
            # read data and filter failed runs
            rates_pre = np.loadtxt("fp_task2c_rates_iteration0.txt")
            rates_pre = rates_pre[rates_pre >= 0]
            gleaks_pre = np.loadtxt("fp_task2c_gleaks_iteration0.txt")
            gleaks_pre = gleaks_pre[gleaks_pre >= 0]
            rates_post = np.loadtxt(f"fp_task2c_rates_iteration{num_iterations-1}.txt")
            rates_post = rates_post[rates_post >= 0]
            gleaks_post = np.loadtxt(
                f"fp_task2c_gleaks_iteration{num_iterations-1}.txt")
            gleaks_post = gleaks_post[gleaks_post >= 0]
            bins = min(len(rates_pre), 20)

            # plot two histograms
            fig = plt.figure()
            ax_rate = fig.add_subplot(221)
            ax_rate.set_title("Uncalibrated firing rates")
            ax_rate.set_xlabel("rates [kHz]")
            ax_rate.set_ylabel("number of neurons")
            ax_rate.hist(rates_pre, bins=bins)

            ax_gleak = fig.add_subplot(222)
            ax_gleak.set_title("Uncalibrated leak conductance")
            ax_gleak.set_xlabel("$g_{leak}$ [S]")
            ax_gleak.set_ylabel("number of neurons")
            ax_gleak.yaxis.tick_right()
            ax_gleak.yaxis.set_label_position("right")
            ax_gleak.hist(gleaks_pre, bins=bins)

            rates_range = [min(np.min(rates_pre), np.min(rates_post)),
                           max(np.max(rates_pre), np.max(rates_post))]
            ax_rate = fig.add_subplot(223, sharex=ax_rate)
            ax_rate.set_title("Calibrated firing rates")
            ax_rate.set_xlabel("rates [kHz]")
            ax_rate.set_ylabel("number of neurons")
            ax_rate.hist(rates_post,
                         bins=bins,
                         range=rates_range)

            gleak_ranges = [min(np.min(gleaks_pre, initial=0.),
                                np.min(gleaks_post, initial=0.)),
                            max(np.max(gleaks_pre, initial=0.),
                                np.max(gleaks_post, initial=0.))]
            ax_gleak = fig.add_subplot(224, sharex=ax_gleak)
            ax_gleak.set_title("Calibrated leak conductance")
            ax_gleak.set_xlabel("$g_{leak}$ [S]")
            ax_gleak.set_ylabel("number of neurons")
            ax_gleak.yaxis.tick_right()
            ax_gleak.yaxis.set_label_position("right")
            ax_gleak.hist(gleaks_post,
                          bins=bins,
                          range=gleak_ranges)

            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9,
                                wspace=0.2, hspace=0.5)
            figname = f'fp_task2c_histo_{time.strftime("%Y%m%d-%H%M%S")}.png'
            plt.savefig(figname)

            print(f"plots saved {figname}")

    Execution

    .. code:: ipython3

        # choose neurons you want to observe
        # the default is range(0,128)
        num_neurons = 100

        # number of iterations for calibration
        num_iterations = 20

        # the leak conductance g_leak is set by I_bias
        # will be adjusted for identical firing rates
        i_bias = [400] * num_neurons

        # calibrate neurons for same firing rate
        calibrate_neurons(i_bias, num_iterations)

        # plot histograms for uncalibrated and calibrated neurons
        plot_histogram(num_iterations)


Exercises
~~~~~~~~~

* Please complete the missing code in the cells above
* Execute all cells, determine sensible parameter characterization ranges
* Document the results that you have obtained and the plots that you have created below

Solution
~~~~~~~~

Please insert your results here...
