A default calibration is generated for every setup every night.
Let us define a function which downloads this nightly calibration and returns the global as well as neuron specific calibration result.

.. code:: ipython3

    from dlens_vx_v2 import hxcomm
    import urllib.request


    # note future versions of pynn_brainscales will support downloading the
    # nightly calibration. For now we have to manually load it.
    def get_nightly_calibration() -> (dict, dict):
        '''
        Download nightly calibration
        '''
        def get_unique_identifier() -> str:
            """
            Get unique identifier of connected setup

            :return: Unique identifier
            """
            with hxcomm.ManagedConnection() as connection:
                identifier = connection.get_unique_identifier()
            return identifier

        identifier = get_unique_identifier()

        download_url = "https://openproject.bioai.eu/data_calibration/" \
                       f"hicann-dls-sr-hx/{identifier}/stable/last-binary/" \
                       "spiking_cocolist.bin"

        # download and save calibration
        contents = urllib.request.urlopen(download_url).read()
        calib_file = "spiking_cocolist.bin"
        with open('spiking_cocolist.bin', 'wb') as f:
            f.write(contents)
        coco = pynn.helper.coco_from_file(calib_file)

        # save calibration data in variables:
        neuron_coco = pynn.helper.filter_atomic_neuron(coco)
        general_coco = pynn.helper.filter_non_atomic_neuron(coco)

        return neuron_coco, general_coco

.. code-block:: ipython3
    :class: test, html-display-none

    # do not download calibration during test but load them directly
    # from disk

    import pynn_brainscales.brainscales2 as pynn
    get_nightly_calibration = pynn.helper.filtered_cocos_from_nightly

We save the nightly calibration in two variables such that we can use it later when we define our neuronal network.

.. code:: ipython3

        neuron_coco, general_coco = get_nightly_calibration()
