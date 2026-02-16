The Exponential Leaky-Integrate-and-Fire Neuron Model
=====================================================

Compared to the standard LIF model, the exponential leaky-integrate-and-fire model adds an exponential term (see `Chapter 5, Section 2 <https://neuronaldynamics.epfl.ch/online/Ch5.S2.html>`_):

.. math::
   :nowrap:

    \begin{align}
           C\frac{\operatorname{d} V}{\operatorname{d} t} &= -g_\text{l} \left( V - E_\text{l} \right)
                   \,+\, g_\text{l} \Delta_\text{T} \operatorname{exp} \left( \frac{V - V_\text{T}}{\Delta_\text{T}} \right)
                   \,+\, I_\text{stim} \,.
    \end{align}

This exponential term introduces strong positive feedback once the membrane potential crosses the exponential threshold :math:`V_\text{T}`.
The steepness of this feedback is controlled by the exponential slope factor :math:`\Delta_\text{T}`.

As before, we use a step current to study the neuron dynamics.
In contrast to earlier experiments, we now vary both the amplitude and the duration of the step current and determine, for each duration, the minimum amplitude that elicits a spike.
We begin by configuring suitable LIF parameters.
Next, we enable the exponential term and adjust the exponential threshold :math:`V_\text{T}`.
Finally, we record a strength-duration curve, which shows the minimum current amplitude required to evoke a spike as a function of stimulus duration.


Setting Everything Up
---------------------

We again establish the connection to the BrainScaleS-2 system and import the required packages.

.. include:: common_quiggeldy_setup.rst

.. code:: ipython3

    %matplotlib inline
    import numpy as np
    import matplotlib.pyplot as plt
    import quantities as pq

    import pynn_brainscales.brainscales2 as pynn


Setting the LIF Parameters
--------------------------

We start by defining an experiment that injects a step current into a single neuron and by configuring the LIF parameters.
You can use the exercises in :ref:`lif_parameters` as a reference and reuse appropriate values.


Exercises
^^^^^^^^^

- **Task 1:**
  Configure a leaky integrator (LI, spiking disabled) neuron with a leak potential around 300 and a membrane time constant of approximately ``100 us``.
  You may reuse values from :ref:`lif_parameters`.
  Note: to obtain long time constants, it may be necessary to set ``leak_enable_division=True``.

- **Task 2:**
  Stimulate the neuron with a step current and plot the membrane voltage.
  Choose an amplitude of 100 and a duration of ``0.1 ms``.
  Leave ``0.2 ms`` before and after the stimulus.
  What do you expect? How will the membrane potential evolve during and after the step?

- **Task 3:**
  Enable spiking and choose a relatively high threshold such that the neuron spikes at a membrane potential of around 750.
  Use a refractory period of 255, a reset potential close to the leak potential, and the maximum reset conductance.
  The currents of the exponential term can become quite high.
  Therefore, we want to disable them during the reset by setting ``refractory_period_enable_pause=True``.
  Why do we intentionally choose such a high threshold?


.. only:: Solution

    **Solution:**

    - **Task 1:**
      Repeat the experiments from :ref:`lif_parameters` and determine suitable parameter values.

    - **Task 2:**
      Reuse the code from :ref:`lif_parameters` to generate a step current.
      Since the membrane time constant is on the same order as the step duration, the membrane potential will not reach steady state.
      Because of the relatively long time constant, the voltage increase appears almost linear compared to earlier experiments.

    - **Task 3:**
      The threshold value can again be determined using the exercises in :ref:`lif_parameters`.
      Alternatively, apply a strong current and adjust the threshold accordingly.
      We choose a high threshold because spike initiation will later be governed by the exponential threshold :math:`V_\text{T}`.
      The hard threshold :math:`V_\text{th}` determines the point where the membrane potential is reset and only has a minimal influence on the dynamics given that the exponential slope factor is high.


Rheobase Current
----------------

Next, we enable the exponential term and configure it such that it just elicits a spike for the chosen parameter set.
This introduces two additional parameters: the exponential slope factor :math:`\Delta_\text{T}` and the exponential threshold :math:`V_\text{T}`.
On BrainScaleS-2, these correspond to ``exponential_i_bias`` and ``exponential_v_exp``.
While ``exponential_i_bias`` mainly controls the slope factor and ``exponential_v_exp`` mainly determines the exponential threshold, both parameters are not fully independent and influence each other on BrainScaleS-2.
In this exercise, we keep ``exponential_i_bias`` fixed and vary ``exponential_v_exp`` until a single spike is elicited.


Exercises
^^^^^^^^^

- **Task 1:**
  Enable the exponential term by setting ``exponential_enable=True`` and choose ``exponential_i_bias=200``.
  Begin with a relatively large value for ``exponential_v_exp`` and decrease it step by step until the neuron reliably emits a spike.
  Then, increase ``exponential_v_exp`` again and determine the largest value for which a spike still occurs.
  Pay attention to the timing of the spike and how it changes as you vary ``exponential_v_exp``.


- **Task 2:**
  Determine the rheobase current, i.e. the minimum constant current amplitude that causes the neuron to spike.
  Use a long stimulus duration of ``10 ms`` and vary the current amplitude accordingly.
  (Optionally) Implement a binary search to find the value automatically.


.. only:: Solution

    **Solution:**

    - **Task 1:**
      Depending on the parameter choice, the neuron may spike even after the current stimulus has been switched off.

    - **Task 2:**

    .. code:: ipython3

        def run_step_current(stim_neuron: pynn.Population,
                             duration: float = 0.2,
                             offset: float = 0.2):
            """
            Stimulate a neuron with a current step.

            :param stim_neuron: Neuron to stimulate. ``constant_current_i_offset`` must be set.
            :param duration: Step current duration in ms.
            :param offset: Time before and after the step (in ms).
            :return: Number of recorded spikes and recorded membrane potentials.
            """
            stim_neuron.record(['v', 'spikes'])

            stim_neuron.set(constant_current_enable=False)

            pynn.run(offset, pynn.RunCommand.APPEND)
            stim_neuron.set(constant_current_enable=True)

            pynn.run(duration, pynn.RunCommand.APPEND)
            stim_neuron.set(constant_current_enable=False)

            pynn.run(offset, pynn.RunCommand.EXECUTE)

            seg = stim_neuron.get_data(clear=True).segments[0]
            pynn.reset()
            return np.sum([len(spikes) for spikes in seg.spiketrains]), seg.irregularlysampledsignals

        pynn.setup()

        pop = pynn.Population(1, pynn.cells.HXNeuron())
        # Note: exact values depend on the hardware setup
        pop.set(
            membrane_capacitance_capacitance=63,
            leak_v_leak=480,
            leak_i_bias=150,
            leak_enable_division=True,
            threshold_enable=True,
            threshold_v_threshold=600,
            reset_v_reset=500,
            reset_i_bias=1022,
            reset_enable_multiplication=True,
            refractory_period_refractory_time=255,
            refractory_period_enable_pause=True,
            exponential_enable=True,
            exponential_i_bias=100,
            exponential_v_exp=580
        )
        pop.record(['spikes'])

        # Adjust this value to determine the rheobase current
        pop.set(constant_current_i_offset=100)

        n_spikes, signals = run_step_current(pop, duration=10)

        fig, ax = plt.subplots()
        for v_mem in signals:
            ax.plot(v_mem.times, v_mem)


Recording a Strength-Duration Curve
-----------------------------------

We now record a strength-duration curve.
For each stimulus duration, we determine the minimum current amplitude that produces a spike.


Exercises
^^^^^^^^^

- **Task 1:**
  What qualitative shape do you expect for the strength-duration curve?

- **Task 2:**
  Modify the code below to record the strength-duration curve.
  Use the previously determined rheobase current as a starting point and increase the amplitude up to 120.
  Limit the amplitude range to 5 values.
  Keep in mind that the hardware is shared among users; therefore, we will implement a more efficient protocol in the next task.
  Does the measured curve roughly match your expectations?

- **Task 3:**
  Improve the efficiency of the code by exploiting the assumption that the required strength increases for shorter durations.
  Start with long durations and increase the amplitude until a spike occurs.
  For the next (shorter) duration, begin the search from the last successful amplitude.
  You can now increase the number of tested durations and amplitudes.
  Does the resulting curve better match your expectations?
  If not, what could explain possible deviations?


.. code:: ipython3

    durations = np.logspace(-1, 0.1, 5)
    amplitudes = np.linspace(50, 120, 5)

    min_amplitudes = []

    # TASK: loop over each duration and determine the minimal amplitude
    # ...

    # TASK: plot the strength-duration curve (minimum amplitude vs. duration)
    # ...


.. only:: Solution

    **Solution:**

    - **Task 1:**
      We expect the required stimulus strength to decrease with increasing duration.

    - **Task 2 & 3:**
      The code looks similar to the previous solution, but we add a loop to determine the minimum amplitudes.

    .. code:: ipython3

        durations = np.logspace(-1, 0.1, 15)
        amplitudes = np.linspace(50, 150, 15)

        min_amplitudes = []

        pynn.setup()

        pop = pynn.Population(1, pynn.cells.HXNeuron())
        # Note: exact values depend on the hardware setup
        pop.set(
            membrane_capacitance_capacitance=63,
            leak_v_leak=480,
            leak_i_bias=150,
            leak_enable_division=True,
            constant_current_i_offset=100,
            threshold_enable=True,
            threshold_v_threshold=600,
            reset_v_reset=500,
            reset_i_bias=1022,
            reset_enable_multiplication=True,
            refractory_period_refractory_time=255,
            refractory_period_enable_pause=True,
            exponential_enable=True,
            exponential_i_bias=100,
            exponential_v_exp=580
        )
        pop.record(['spikes'])

        # Go in reverse order to reduce the number of experiments needed:
        # we will not test all amplitudes but only those equal to
        # or higher than the last successful one.
        for duration in durations[::-1]:
            value_found = False
            for amplitude in amplitudes:
                # amplitudes should increase with decreasing duration
                if len(min_amplitudes) > 0 and amplitude < min_amplitudes[-1]:
                    continue

                pop.set(constant_current_i_offset=amplitude)
                n_spikes, _ = run_step_current(pop, duration=duration)

                if n_spikes > 0:
                    min_amplitudes.append(amplitude)
                    value_found = True
                    break

            if not value_found:
                min_amplitudes.append(np.nan)

        # we have recorded the amplitudes in reverse order
        min_amplitudes.reverse()

        fig, ax = plt.subplots()
        ax.plot(durations, min_amplitudes)
        ax.set_xlabel("Duration / us")
        ax.set_ylabel("Strength / a.u.")
