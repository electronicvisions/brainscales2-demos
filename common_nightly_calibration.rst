A default calibration is generated for every setup every night.
We save the nightly calibration in a variable such that we can use it later when we define our experiment.

.. code:: ipython3

    from _static.common.helpers import get_nightly_calibration
    calib = get_nightly_calibration()
