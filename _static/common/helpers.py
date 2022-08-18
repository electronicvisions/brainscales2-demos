import os
import urllib.request

import hashlib
import pandas as pd

def in_collaboratory():
    return os.environ.get('JUPYTERHUB_USER') is not None

if in_collaboratory():
    # check for EBRAINS kernel version prior to pynn_brainscales import
    expected_kernel = 'EBRAINS-22.07'
    actual_kernel = os.environ.get('LAB_KERNEL_NAME', None)
    if actual_kernel is None:
        raise RuntimeError(
            "Could not identify EBRAINS kernel (probably too old version). " +
            f"Please select the appropriate kernel {expected_kernel}.")
    elif actual_kernel != expected_kernel:
        raise RuntimeError(
            f"EBRAINS kernel mismatch. Expected: {expected_kernel} " +
            f"Actual: {actual_kernel}. Please select the " +
            "appropriate kernel.")

from dlens_vx_v3 import hxcomm
import pynn_brainscales.brainscales2 as pynn


def setup_hardware_client():
    if not in_collaboratory():
        return

    # setup quiggeldy enviroment
    setup_url = 'https://brainscales-r.kip.uni-heidelberg.de:7443/nmpi/' \
                'quiggeldy_setups.csv'
    quiggeldy_setups = pd.read_csv(setup_url, dtype=str)

    os.environ['QUIGGELDY_ENABLED']='1'

    username = os.environ.get('JUPYTERHUB_USER')
    assert username is not None
    os.environ['QUIGGELDY_USER_NO_MUNGE'] = username

    index = int(hashlib.sha256(username.encode()).hexdigest(), 16) \
        %  len(quiggeldy_setups)
    my_setup = quiggeldy_setups.iloc[index]
    os.environ['QUIGGELDY_IP'] = my_setup['Host']
    os.environ['QUIGGELDY_PORT'] = my_setup['Port']


def get_nightly_calibration(filename='spiking_cocolist.pbin'):
    '''
    Get the nightly deployed calibration.

    If the code is executed in the EBrains in the collabatory the
    calibration is downloaded otherwise it is expected to be present
    locally.

    '''
    if in_collaboratory():
        with hxcomm.ManagedConnection() as connection:
            identifier = connection.get_unique_identifier()

        # download calibration file
        folder =  "latest"
        download_url = "https://openproject.bioai.eu/data_calibration/" \
                       f"hicann-dls-sr-hx/{identifier}/stable/{folder}" \
                       f"/{filename}"
        contents = urllib.request.urlopen(download_url).read()

        path_to_calib = filename
        with open(path_to_calib, 'wb') as f:
            f.write(contents)
    else:
        calib_path = pynn.helper.nightly_calib_path().parent
        path_to_calib =  calib_path.joinpath(filename)

    chip = pynn.helper.chip_from_file(path_to_calib)

    return chip
