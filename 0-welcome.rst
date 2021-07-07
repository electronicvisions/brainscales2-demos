Welcome to the BrainScaleS-2 Tutorial
=====================================

.. image:: _static/bss2.jpg

Hello and welcome to this tutorial that will interactively guide you through your first experiments on the BrainScaleS-2 system!

You will learn the basic tools for running experiments on the BrainScaleS-2 platform.
For inspiration, please refer to the following list for examples of previous scientific work done on the system:

* `Versatile emulation of spiking neural networks on an accelerated neuromorphic substrate <https://ieeexplore.ieee.org/document/9180741>`_
* `Surrogate gradients for analog neuromorphic computing <https://arxiv.org/abs/2006.07239>`_
* `hxtorch: PyTorch for BrainScaleS-2 – Perceptrons on Analog Neuromorphic Hardware <https://link.springer.com/chapter/10.1007/978-3-030-66770-2_14>`_
* `Control of criticality and computation in spiking neuromorphic networks with plasticity <https://www.nature.com/articles/s41467-020-16548-3>`_
* `Demonstrating Advantages of Neuromorphic Computation: A Pilot Study <https://www.frontiersin.org/articles/10.3389/fnins.2019.00260>`_
* `Fast and deep: energy-efficient neuromorphic learning with first-spike times <https://arxiv.org/abs/1912.11443>`_
* `Inference with Artificial Neural Networks on Analog Neuromorphic Hardware <https://link.springer.com/chapter/10.1007/978-3-030-66770-2_15>`_
* `Spiking neuromorphic chip learns entangled quantum states <https://arxiv.org/abs/2008.01039>`_
* `Structural plasticity on an accelerated analog neuromorphic hardware system <https://www.sciencedirect.com/science/article/pii/S0893608020303555>`_


In this session, we will cover the following topics:

.. toctree::
   :maxdepth: 1

   1-single_neuron
   2-superspike
   3-hagen_intro
   4-hagen_properties
   5-plasticity_rate_coding

In this section of the tutorial, we will go through the technical details and make sure that you are correctly set up for accessing our hardware resources.


Jupyter Notebook Setup
----------------------
We have set up a Jupyter instance running on a local compute cluster in Heidelberg, which you can access via https://juphub.bioai.eu.
To login, you will need a valid *EBRAINS* account.
Once logged in, you can start a personal notebook server, which will contain two folders needed for this tutorial:

* ``tutorials``:
  Your personal copy of the prepared tutorial.
  This is the place to run and edit experiments at during this session.
* ``tutorials-read-only``:
  Read-only reference of the original notebooks.
  You may refer back to these in case anything should stop working for you during the sesssion.


Shared Hardware Ressources
--------------------------
All notebooks used in this tutorial are running on a total of eight BrainScaleS-2 ASICs.
We utilize the intrinsic speed of the system to offer you an interactive experience that is as smooth as possible even though multiple participants will access the same chip at any given point in time.

This process is hidden by a custom microscheduler (*quiggeldy*), a conceptual view of which you can see in the following figure.
The actual hardware execution time has been colored in blue.

.. image:: _static/daas_multi.png
    :width: 80%
    :align: center

Please note that the hardware performance you will experience is affected by other users in this tutorial and can not be percieved as an accurate representation of the expected performance for single-user workloads.


Final test: Hardware Execution
------------------------------
Before we start with the actual tutorial, we'd like to ensure that you are correctly set up for running experiments on the BrainScaleS-2 platform.
To do so, simply run the following minimal pyNN-experiment.
It should terminate without errors.

.. code-block:: python3

    import pynn_brainscales.brainscales2 as pynn

    pynn.setup()
    neurons_1 = pynn.Population(2, pynn.cells.HXNeuron())
    neurons_2 = pynn.Population(3, pynn.cells.HXNeuron())
    pynn.Projection(neurons_1, neurons_2, pynn.AllToAllConnector())
    pynn.run(0.2)
    pynn.end()

