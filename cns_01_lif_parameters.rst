.. _lif_parameters:

Parameters of a Leaky-Integrate-and-Fire Neuron Model
=====================================================

In this notebook, we explore the leaky-integrate-and-fire (LIF) neuron model.
You will learn how to configure the model on BrainScaleS-2, how its parameters influence the neuronal dynamics, and how hardware parameters map to the abstract model parameters.

We begin by disabling the spiking mechanism and studying a pure leaky integrator.
This allows us to determine its membrane time constant and to establish a mapping between hardware parameters and model parameters.

The dynamics of the LIF model are described by:

.. math::
   :nowrap:

    \begin{align}
           C\frac{\operatorname{d} V}{\operatorname{d} t} &= -g_\text{l} \left( V - E_\text{l} \right)
                   \,+\, I_\text{stim} \, ,
    \end{align}

where :math:`E_\text{l}` is the leak potential, :math:`C` the membrane capacitance, and :math:`g_\text{l}` the leak conductance.
:math:`I_\text{stim}` denotes an external input current.
When the membrane voltage :math:`V` reaches the threshold :math:`V_\text{th}`, the neuron is reset to the reset potential :math:`V_\text{r}` and remains there for the refractory period :math:`\tau_\text{ref}`.
For further details, see `Chapter 1, Section 3 of Neuronal Dynamics <https://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_.


Setting Everything Up
---------------------

Before starting the experiments, we establish the connection to the BrainScaleS-2 system:

.. include:: common_quiggeldy_setup.rst

Next, we import useful Python packages and the PyNN interface, which we use to define and run our experiments.

.. code:: ipython3

    %matplotlib inline
    from typing import Tuple
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    import quantities as pq

    import pynn_brainscales.brainscales2 as pynn


.. _the_leak_potential:

The Leak Potential
------------------

In this example, we demonstrate how to configure a leaky integrator neuron on BrainScaleS-2.
Each experiment defined with PyNN starts with a call to ``pynn.setup()``.
After that, we can define neuron populations, set parameters, create connections, and specify what variables should be recorded.

In the given tutorials, we use cells of type ``HXNeuron`` which represent neuron circuits on BrainScaleS-2.
In the following we explore how its parameters are related to the parameters of the LIF model.
By default, all parameters of the ``HXNeuron`` are initialized to zero.
As a result, the spiking mechanism is initially disabled and we have to manually set the neuron parameters.

We set the membrane capacitance, leak conductance, and leak potential, record the membrane voltage, and run the experiment for ``0.1 ms``.
Although this appears short compared to biological time scales, BrainScaleS-2 operates at an accelerated speed.
In these experiments, we use a speed-up factor of 1000, so time constants are in the range microseconds rather than milliseconds; and experiment runtimes in the range of milliseconds instead of seconds.

.. code:: ipython3

    pynn.setup()

    pop = pynn.Population(1, pynn.cells.HXNeuron())
    pop.set(
        # Membrane capacitance, range 0-63
        membrane_capacitance_capacitance=63,
        # Leak potential, range: 0-1022
        leak_v_leak=500,
        # Leak conductance:  0-1022
        leak_i_bias=500)
    pop.record('v')

    pynn.run(0.1)  # time in ms

After execution, the recorded data can be retrieved from the population and plotted.

.. code:: ipython3

    v_mem = pop.get_data(clear=True).segments[0].irregularlysampledsignals[0]

    fig, ax = plt.subplots()
    ax.plot(v_mem.times, v_mem)
    ax.set_ylim(0, 1022)
    ax.set_xlabel("Time / ms")
    ax.set_ylabel("V / a.u.")

The membrane voltage is measured using an analog-to-digital converter (ADC).
For simplicity, we use the raw digital values in this tutorial and do not convert them to SI units.


Exercises
^^^^^^^^^

- **Task 1:**
  Modify the neuron parameters and observe how they influence the recorded membrane potential.
  Which parameters have the strongest effect?
  You may need to adjust the y-axis limits to better visualize the changes.

- **Task 2:**
  Keep the capacitance at its maximum value and set the leak conductance to a medium value.
  Sweep ``leak_v_leak`` over ten values in the range 0–1022 and plot the recorded mean membrane potential against ``leak_v_leak``.
  Refer to the note below for running multiple experiments in a loop.
  What relationship do you observe between ``leak_v_leak`` and the mean membrane potential?

.. only:: Solution

    **Solution:**

    - **Task 1:**
      The mean membrane potential is primarily determined by ``leak_v_leak``.
      The parameter ``leak_i_bias`` also influences the potential, but to a lesser extent.

    - **Task 2:**
      We observe an approximately linear relationship, with saturation near the parameter boundaries.

    .. code:: ipython3

        pynn.setup()

        pop = pynn.Population(1, pynn.cells.HXNeuron())

        leak_v_leak_values = np.linspace(0, 1022, 10)
        mean_potentials = []

        for leak_v_leak in leak_v_leak_values:
            pop.set(
                membrane_capacitance_capacitance=63,
                leak_v_leak=leak_v_leak,
                leak_i_bias=500)
            pop.record('v')
            pynn.run(0.1)
            v_mem = pop.get_data(clear=True).segments[0].irregularlysampledsignals[0]
            mean_potentials.append(v_mem.mean())
            pynn.reset()

        fig, ax = plt.subplots()
        ax.plot(leak_v_leak_values, mean_potentials)
        ax.set_ylim(0, 1022)
        ax.set_xlabel("leak_v_leak")
        ax.set_ylabel("Mean Potential / a.u.")


Note: Loops
^^^^^^^^^^^

If you want to repeat experiments with the same network topology while changing only parameters, you can use ``pynn.reset()`` to reset the experiment time.
Each call to ``pynn.reset()`` starts a new data segment.
When calling ``pynn.setup()``, ``pynn.end()``, or retrieving data with ``clear=True`` in ``Population.get_data()``, these segments are removed.

.. code:: ipython3

    pynn.setup()

    pop = pynn.Population(1, pynn.cells.HXNeuron())
    pop.set(
        membrane_capacitance_capacitance=63,
        leak_v_leak=500,
        leak_i_bias=500)
    pop.record('v')

    for _ in range(3):
        pynn.run(0.1)
        pynn.reset()  # reset the experiment time and start a new segment

    fig, ax = plt.subplots()
    for n_run, seg in enumerate(pop.get_data(clear=True).segments):
        v_mem = seg.irregularlysampledsignals[0]
        ax.plot(v_mem.times, v_mem, label=f"Run {n_run}")

    ax.legend()
    ax.set_ylim(0, 1022)
    ax.set_xlabel("Time / ms")
    ax.set_ylabel("V / a.u.")


The Leak Conductance
--------------------

Next, we measure the membrane time constant :math:`\tau_\text{mem} = C / g_\text{l}` as a function of the hardware parameter ``leak_i_bias``.
To do this, we inject a current step and fit an exponential to the decay of the membrane potential after the current is switched off.

The following example demonstrates how to generate a current step on BrainScaleS-2:

.. code:: ipython3

    pynn.setup()

    pop = pynn.Population(1, pynn.cells.HXNeuron())
    pop.set(
        membrane_capacitance_capacitance=63,
        leak_v_leak=500,
        leak_i_bias=500,
        # Strength of current input, range: 0-1022
        constant_current_i_offset=500)
    pop.record('v')

    offset = 0.2  # ms, offset before/after current step
    duration = 0.1  # ms, duration of current step

    # Disable/enable the current input at specific points in time
    pop.set(constant_current_enable=False)

    pynn.run(offset, pynn.RunCommand.APPEND)
    pop.set(constant_current_enable=True)

    pynn.run(duration, pynn.RunCommand.APPEND)
    pop.set(constant_current_enable=False)

    pynn.run(offset, pynn.RunCommand.EXECUTE)

    signals = pop.get_data(clear=True).segments[0].irregularlysampledsignals
    fig, ax = plt.subplots()
    for v_mem in signals:
        ax.plot(v_mem.times, v_mem)

    ax.set_ylim(0, 1022)
    ax.set_xlabel("Time / ms")
    ax.set_ylabel("V / a.u.")

Using ``pynn.run(..., pynn.RunCommand.APPEND)``, we advance the simulation time without executing it on the hardware.
This allows us to enable or disable the current dynamically.
The experiment is executed with ``pynn.run(..., pynn.RunCommand.EXECUTE)``.
Each part of the experiment is stored in separate signals, which is reflected by different colors in the plot.

We can fit an exponential to the decay phase using ``scipy.optimize.curve_fit()``:

.. code:: ipython3

    def exponential(t: float,
                    amplitude: float,
                    offset: float,
                    tau: float) -> float:
        """
        Exponential decay.

        :param t: Time t.
        :param amplitude: Amplitude (A) of the exponential.
        :param offset: Constant offset (C) of the value.
        :param tau: Time constant (tau) of the decay.
        :return:  A * exp(-t/tau) + C.
        """
        return amplitude * np.exp(- t / tau) + offset


    def fit_exponential(times: np.ndarray, values: np.ndarray) -> Tuple[float]:
        """
        Fit an exponential to the given data.

        The exponential decay should start at the beginning of the recording.

        :param times: Recording times.
        :param values: Recorded values.
        :return: Fit parameters: amplitude, offset, tau.
        """

        norm_times = times - times[0]

        offset_p0 = values[-1]
        norm_values = values - offset_p0

        # Estimate tau by taking two values
        idx_start = 0
        idx_stop = min(60, len(norm_times))
        if np.abs(norm_values[idx_stop] - norm_values[idx_start]) < 10:
            tau_p0 = 0.1
        else:
            tau_p0 = - (norm_times[idx_stop] - norm_times[idx_start]) \
                / np.log(norm_values[idx_stop] / norm_values[idx_start])

        p0 = (values[0] - offset_p0, offset_p0, tau_p0)
        return scipy.optimize.curve_fit(exponential,
                                        norm_times,
                                        values,
                                        p0=p0)[0]


    sig = signals[-1]
    # take the `magnitude` and `flatten` for easier processing
    fit_params = fit_exponential(sig.times.magnitude, sig.magnitude.flatten())

    fig, ax = plt.subplots()
    for v_mem in signals:
        ax.plot(v_mem.times, v_mem)

    norm_times = sig.times.magnitude - sig.times.magnitude[0]
    ax.plot(sig.times, exponential(norm_times, *fit_params), c='w', lw=3)
    ax.plot(sig.times, exponential(norm_times, *fit_params), c='k', lw=2)

    ax.set_ylim(0, 1022)
    ax.set_xlabel("Time / ms")
    ax.set_ylabel("V / a.u.")

    print(f"Fitted time constant: {fit_params[-1]} ms")


Exercises
^^^^^^^^^

- **Task 1:**
  Adjust ``leak_v_leak`` so that the membrane voltage is approximately 300 without current.
  Then choose ``constant_current_i_offset`` such that the voltage increases by about 100 during the current step.

- **Task 2:**
  Measure the membrane time constant for ten values of ``leak_i_bias`` in the range 10–1022.
  What relationship do you expect?
  Try plotting the reciprocal time constant to better visualize the dependency.
  If the results deviate from your expectations, what might explain this?

- **Task 3:**
  How does the maximum membrane voltage during the current step depend on the leak conductance?
  Derive your expectation from the model equation before performing the experiment.

.. only:: Solution

    **Solution:**

    - **Task 1:**
      The appropriate value for ``leak_v_leak`` can be determined from the previous plots.

    - **Task 2:**
      We expect :math:`1 / \tau` to depend linearly on the leak conductance :math:`g_\text{l}`.
      Plotting the reciprocal highlights this relationship.
      The parameterization provides finer resolution for small time constants (higher conductances), which explains deviations from strict linearity.

    .. code:: ipython3

        def run_step_current(stim_neuron: pynn.Population,
                             duration: float = 0.2,
                             offset: float = 0.2):
            """
            Stimulate a neuron with a current step.

            :param stim_neuron: Neuron to stimulate. ``constant_current_i_offset`` has
                to be set.
            :param duration: Step current duration in ms.
            :param offset: Time before/after the step current (in ms).
            :return: Recorded membrane potentials for the different parts of
                the experiment.
            """

            stim_neuron.record('v')

            stim_neuron.set(constant_current_enable=False)

            pynn.run(offset, pynn.RunCommand.APPEND)
            stim_neuron.set(constant_current_enable=True)

            pynn.run(duration, pynn.RunCommand.APPEND)
            stim_neuron.set(constant_current_enable=False)

            pynn.run(offset, pynn.RunCommand.EXECUTE)

            signals = stim_neuron.get_data(clear=True).segments[0].irregularlysampledsignals
            pynn.reset()
            return signals

        pynn.setup()

        pop = pynn.Population(1, pynn.cells.HXNeuron())

        leak_i_bias_values = np.linspace(10, 1022, 10)
        tau = []

        for leak_i_bias in leak_i_bias_values:
            pop.set(
                membrane_capacitance_capacitance=63,
                leak_v_leak=500,
                leak_i_bias=leak_i_bias,
                constant_current_i_offset=500)
            pop.record('v')
            sigs = run_step_current(pop)
            tau.append(fit_exponential(sigs[-1].times.magnitude,
                                       sigs[-1].magnitude.flatten())[-1])

        fig, ax = plt.subplots()
        ax.plot(leak_i_bias_values, 1 / np.array(tau))
        ax.set_xlabel("leak_i_bias")
        ax.set_ylabel("1 / tau / (1 / us)")

    - **Task 3:**
      Setting the derivative to zero in the model equation shows that the steady-state voltage during the current step is proportional to :math:`I / g_\text{l}`.
      This can be verified experimentally by measuring the voltage increase during the step.


Enabling Firing
---------------

Finally, we enable the spiking mechanism.
On BrainScaleS-2, the reset is not instantaneous.
Instead, when the membrane reaches the threshold :math:`V_\text{th}`, it is connected to the reset potential :math:`V_\text{r}` via a conductance :math:`g_\text{r}`.
For the following experiments, we set the reset conductance and refractory period to their maximum values.
In addition to the membrane voltage, we now also record spikes.
Once again, a spike train is generated for each part of the experiment.

.. code:: ipython3

    pynn.setup()

    pop = pynn.Population(1, pynn.cells.HXNeuron())
    pop.set(
        membrane_capacitance_capacitance=63,
        leak_v_leak=500,
        leak_i_bias=500,
        constant_current_i_offset=500,
        threshold_enable=True,
        # Threshold potential, range: 0-1022
        threshold_v_threshold=800,
        # Reset potential, range: 0-1022
        reset_v_reset=300,
        # Reset conductance, range: 0-1022
        reset_i_bias=1022,
        # Increase reset conductance
        reset_enable_multiplication=True,
        # Refractory time (counter), range: 10-255
        refractory_period_refractory_time=255,
        )
    pop.record(['v', 'spikes'])

    offset = 0.2  # ms, offset before/after current step
    duration = 0.1  # ms, duration of current step

    pop.set(constant_current_enable=False)

    pynn.run(offset, pynn.RunCommand.APPEND)
    pop.set(constant_current_enable=True)

    pynn.run(duration, pynn.RunCommand.APPEND)
    pop.set(constant_current_enable=False)

    pynn.run(offset, pynn.RunCommand.EXECUTE)

    data = pop.get_data(clear=True).segments[0]
    signals = data.irregularlysampledsignals
    spiketrains = data.spiketrains
    fig, ax = plt.subplots()
    for v_mem in signals:
        ax.plot(v_mem.times, v_mem)

    for spiketrain in spiketrains:
        for spike in spiketrain:
            ax.axvline(spike, ymin=0.9, ymax=0.95, c='k')

    ax.set_ylim(0, 1022)
    ax.set_xlabel("Time / ms")
    ax.set_ylabel("V / a.u.")


Exercises
^^^^^^^^^

- **Task 1:**
  Adjust the parameters such that the membrane time constant is approximately ``10 us``.

- **Task 2:**
  Decrease the threshold until spikes occur.
  How does the threshold affect the firing frequency?

- **Task 3:**
  Modify the reset potential.
  How does it influence the firing frequency?

- **Task 4 (optional):**
  As in Task 2 of :ref:`the_leak_potential`, sweep ``threshold_v_threshold`` and ``reset_v_reset`` to determine how the hardware parameters map to the measured threshold and reset voltages.
  Ensure that the current input is sufficiently strong when testing high thresholds.
  Instead of fitting an exponential, extract the maximum or minimum voltage during the step to estimate the threshold or reset value.
  You may extend the duration of the current step to obtain multiple spikes per trial.


.. only:: Solution

    **Solution:**

    - **Task 1:**
      The appropriate value for ``leak_i_bias`` can be read from the previously obtained time-constant plot.

    - **Task 2:**
      The firing frequency increases as the threshold is lowered.

    - **Task 3:**
      The firing frequency increases as the reset potential is raised.

    - **Task 4:**
      Reuse the experimental protocol from earlier.
      Sweep over ``threshold_v_threshold`` (with sufficiently strong input) or ``reset_v_reset``.
      Instead of computing the time constant, extract the peak or minimum voltage during the step to estimate the effective threshold or reset potential.
