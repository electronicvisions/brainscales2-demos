Biological background
=====================

.. image:: _static/fp/cajal_cortex_drawings.png
   :width: 40%
   :align: center



Computational neuroscience
~~~~~~~~~~~~~~~~~~~~~~~~~~

The central neural system is highly complex and responsible for many functions in diverse life forms.
From processing sensorial data, coordinating movement or having "thoughts", all this is managed by this system.
One building block of this system is the neuron.
In this work we will only dip our toes in the deep water of neuromorphic engineering, limiting ourselves to a few models of neurons and synapses, learning behaviors of simple models and even dive a little deeper in the end.

We start with a model of an ideal spiking neuron (`Gerstner (2014) <https://courses.edx.org/c4x/EPFLx/BIO465x/asset/nd_ch1.pdf>`_); the name will become obvious after the introduction of its basic behaviors.

Neurons in biology
^^^^^^^^^^^^^^^^^^

A neuron can be separated into three major parts which serve different purposes:
The central unit of the cell is called soma, that receives 'input' from the dendritic tree, basically branched out wires extending around the soma.
The 'input' arrives at synapses in the form of (chemical) neurotransmitters that are released following activity of a presynaptic neuron.
The activity arrives in the form of so-called action potentials or spikes, (electrical) signals that are emitted at a soma and travel through the third part of a neuron, the axon.
The neurotransmitters lead to voltage differences at the dendrites, again an electrical signal, which in turn affect the soma that can send out spikes via its axon when a sufficient amount of input has accumulated.

Let's take a closer look at the physical behavior of a cell, especially the membrane potential of the soma.
In the absence of any stimulation, we observe a negative resting voltage of approximately :math:`-65` mV, a strongly negative polarized state.
A stimulation happening through the dendrites changes the membrane potential.
This can be formalized as follows:
At a given time :math:`t` the membrane potential of neuron :math:`i` is :math:`u_i(t)`.
When the neuron is not stimulated, for all times smaller :math:`0`, the potential is :math:`u_i(t<0) = u_\text{rest}`.
At :math:`t=0` a stimulation occurs in form of a spike  which is triggered by a presynaptic cell.
We define

.. math::
    \epsilon_{ij}(t) := u_i(t) - u_\text{rest}


as the postsynaptic potential (PSP).
To indicate the direction of change we define further

.. math::

    \epsilon_{ij} = u_i(t) - u_\text{rest}
    \begin{cases}
    >0 & \quad \text{excitatory PSP} \\
    <0 & \quad \text{inhibitory PSP}
    \end{cases}


Typically, the potential of the membrane does not stay at the same level, it decays towards the resting potential :math:`u_\text{rest}`.

Now let's assume there are 2 presynaptic neurons :math:`j=1,2`.
These neurons emit signals at respective times :math:`t_j^{(m)}` where :math:`m` states the :math:`m`-th signal peak coming from that neuron.
At each incoming signal peak, charges are deposited at the membrane and the potential rises.
When many excitatory inputs have arrived at the neuron, the membrane potential can reach a critical value, triggering the neuron to fire.
A sharp rise of the membrane potential (reaching a voltage of about :math:`+100 mV`) is observed and the potential propagates down the axon.
The neuron is said to have 'spiked' and the pulse is referred to as a spike.

After such a pulse, the membrane potential drops below the resting voltage and later it returns back.
This is called hyperpolarization or spike-afterpotential.
In that time the neuron can't be stimulated.
The trajectory of such an event can be observed in following image.

.. image:: _static/fp/fp_introduction_psp_staking.png
   :width: 40%
   :align: center


On the x axis is the time while on the y axis is the membrane potential.
At given times :math:`t_i^{(n)}` the :math:`n\text{th}` spike from neuron :math:`i` arrives at our observed neuron.
Each spike leads to a rise of the membrane potential :math:`u`.
The dotted line indicates the assumed path if there wouldn't be a change in current.
In this instance :math:`t_2^{(2)}` is enough to cross :math:`\vartheta` (threshold) and the neuron by itself fires.
In the case where the threshold is not reached, i.e., when only a few, weak spikes appear, the voltage behaves as the sum of the individual PSPs:

.. math::
    :label: eq:psp_stacking

    u_i(t) \approx \left[
    \sum_j \sum_f \epsilon_{ij} \left(t - t_j^{(f)}\right)
    \right] + u_\text{rest}

This is also called PSP-Staking (Image was taken from `Gerstner et al. 2014, Chapter 1.2 <ttps://courses.edx.org/c4x/EPFLx/BIO465x/asset/nd_ch1.pdf>`_)

In the next step we want to find a way to make the model, i.e., the equations, more concrete, so it is easy to implement it on a neuromorphic substrate.

Modelling neuronal behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Leaky integrate-and-fire (LIF) model
+++++++++++++++++++++++++++++++++++++

The neuron is driven by biochemical and bioelectrical principles, involving various interactions (for a short overview see `Purves (2009) <http://www.scholarpedia.org/article/Neuroscience>`_).
We aim to find a suitable model that describes the basic behavior of neurons without necessarily incorporating all biological aspects.
The goal is to obtain a somewhat similar behavior as described in the previous section.
The following constraints are applied:
The spikes always have to have the same shape, i.e., the shape does not contain any information.
Instead, all information will be coded in the timings of the spikes.
Further, membrane potential processes the information.
Another feature, that needs to be modelled is when :math:`u_i(t)` reaches a critical voltage threshold :math:`\vartheta`, a spike has to be initialized, causing this neuron "to fire" at this time :math:`t_i^{(m)}`.
Such a model was proposed by `Lapique (1907) <https://link.springer.com/article/10.1007/s00422-007-0189-6>`_ and is called leaky fire-and-integrate (LIF).

Essentially, a cell membrane acts as a good insulator.
When a current :math:`I(t)` arrives at the membrane, additional charge is deposited.
This behavior is similar to a capacitor and therefore we abstract a cell membrane by it with a certain capacitance :math:`C`.
As previously discussed, the membrane potential decreases over time; therefore, the charge leaks.
This can be emulated with a resistance :math:`R`.
In addition we require a source to define a resting potential.
This completes the basic circuitry for a neuron model:

.. image:: _static/fp/fp_circuit.png
   :width: 30%
   :align: center

If we analyze the electrical circuit, we can find a differential equation describing the behavior of the capacitor voltage:

.. math::
    :label: eq:lif
    
    \tau_m \frac{\mathrm{d} u_i(t)}{\mathrm{d} t} = - \left[u_i(t) -u_\text{rest} \right] + R \cdot I(t)



Here, :math:`\tau_m = R \cdot C` is also called the *membrane time constant*, and the index :math:`i` refers to the i-th neuron.
:math:`I(t)` in this equation represents a time dependent current flow onto (excitatory) or away from (inhibitory) the membrane.
In neuroscience this equation, which describes a leaky integrator, is the equation of a passive neuron.
Currently, it fulfills the requirement of integrating incoming spikes (see equation :eq:`eq:psp_stacking`), but it lacks an active part in the form of a spiking mechanism.
For the basic model, we define a threshold value :math:`\vartheta`.
When this value is crossed from below, an additional circuit emits a voltage spike that propagates to all connected neurons.
At the same time, the potential of the capacitance is clamped to a defined value :math:`u_\text{reset}` and kept at this level for the *refractory period* :math:`\tau_\text{r}`.

Adaptive exponential (AdEx) model
++++++++++++++++++++++++++++++++++

The LIF-model captures only some basic dynamics of a real neuron.
Various models aim to enhance this understanding.
In `Brette (2005) <https://journals.physiology.org/doi/full/10.1152/jn.00686.2005>`_ an improved LIF-model is presented.
The main additions are an exponential and an adaptation term.
This is required to process high frequent synaptic input and model the spike initiation accurately.
Further, a recovery variable is introduced to capture adaptation and resonance properties.

.. math::
        :label: eq:adex

        \tau_m \frac{\mathrm{d}u_i}{\mathrm{d}t} &= - \left( u_i - u_\text{rest} \right) + R \left( I_\text{stim} - I_\text{ad} \right) + \Delta_\text{T} \mathrm{exp}\left( \frac{u_i-u_T}{\Delta_\text{T}} \right) \\ 
        \tau_w \frac{\mathrm{d}I_\text{ad}}{\mathrm{d}t} &= a \left( u_i - u_\text{rest} \right) - I_\text{ad}

The equation above is arranged in such a way that the extension to the LIF-equation :eq:`eq:lif` is easily visible.
As new terms are introduced: 
:math:`I_\text{ad}` which is the adaptation current,
:math:`\Delta_\text{T}` is the threshold slope factor,
:math:`u_\text{T}` is the effective threshold potential.
Further, a second equation is introduced to describe the dynamics of the adaptation.
For this, a conductance :math:`a` and the time constant for the adaptation current :math:`\tau_w` is required.
Another modification is the reset condition.
While previously only the neuron was set to a reset potential, now the adaptation has to be modified as well.
The action is now

.. math::
      \text{if } u_i>\vartheta \text{ then } \begin{cases}
          u_i \rightarrow u_\text{reset} \\
          I_\text{ad} \rightarrow I_\text{ad} + b
      \end{cases}

Here, an additional variable is used,  the spike triggered adaptation :math:`b`.

This model is called AdEx due to the exponential term.
Depending on the parametrisation, it can describe different neuron types and model more sophisticated behaviors.
