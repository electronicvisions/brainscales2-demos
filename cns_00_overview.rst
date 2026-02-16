Introduction to Computational Neuroscience with BrainScaleS-2
=============================================================

In this collection of tutorial notebooks, we aim to introduce computational neuroscience together with the neuromorphic BrainScaleS-2 system.
The exercises follow the book "Neuronal Dynamics" by Gerstner et al. and the accompanying Python exercises (`online version <https://neuronaldynamics.epfl.ch/>`_).


How to Execute the Notebooks
----------------------------

In order to execute the notebooks, you need access to the BrainScaleS-2 system.
We provide this access via the `EBRAINS Platform <https://wiki.ebrains.eu>`_.
You can register there free of charge and create your own "collaboratory."
In this collaboratory, you can start a JupyterLab session and directly clone the notebooks from our `GitHub repository <https://github.com/electronicvisions/brainscales2-demos/tree/jupyter-notebooks-experimental>`_:

.. code-block:: bash

   !git clone https://github.com/electronicvisions/brainscales2-demos.git --branch jupyter-notebooks-experimental


Hardware Execution
------------------

In order to access BrainScaleS-2, we have to set up a connection to the hardware.
To set up a connection to a shared BrainScaleS-2 system, we can execute the following command:

.. include:: common_quiggeldy_setup.rst

We will include this command in each of the notebooks.


Before we start with the tutorial, we want to make sure that the setup was successful:

.. code-block:: python3

    import pynn_brainscales.brainscales2 as pynn

    pynn.setup()
    neurons_1 = pynn.Population(2, pynn.cells.HXNeuron())
    neurons_2 = pynn.Population(3, pynn.cells.HXNeuron())
    pynn.Projection(neurons_1, neurons_2, pynn.AllToAllConnector())
    pynn.run(0.2)
    pynn.end()
    print("Everything is set! We are ready for the tutorials.")


Exercise Overview
-----------------

In the following, we give a quick overview of the different tutorial notebooks.


1. Parameters of a Leaky-Integrate-and-Fire Neuron Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this tutorial, we introduce the leaky-integrate-and-fire (LIF) neuron model and explore how its parameters influence its dynamics.
We start with a leaky integrator (a neuron that does not elicit action potentials) and stimulate it with a step current in order to explore the time scale on which it integrates information.
Next, we enable the spiking mechanism and see how the different parameters influence the firing behavior of the neuron.
Apart from the LIF neuron, this tutorial also shows how we define experiments for the BrainScaleS-2 system and how its parameters relate to the parameters of the model.

3. The Exponential Leaky-Integrate-and-Fire Neuron Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this notebook, we introduce an exponential term to our LIF model.
In a so-called strength-duration curve, we illustrate how the strength and duration of a step current influence the spiking behavior of our neuron model.


Table of Contents
^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   cns_01_lif_parameters
   cns_03_exponential_lif
