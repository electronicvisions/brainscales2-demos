import os
import urllib.request
from typing import Optional
from pathlib import Path
import shutil

import hashlib
import pandas as pd

from _static.common.collab_helpers import in_collaboratory, check_kernel

if in_collaboratory():
    # check for EBRAINS kernel version prior to pynn_brainscales import
    check_kernel()

from dlens_vx_v3 import hxcomm
import pynn_brainscales.brainscales2 as pynn


def setup_hardware_client():
    if not in_collaboratory():
        return

    # setup quiggeldy enviroment
    setup_url = 'https://brainscales-r.kip.uni-heidelberg.de:7443/nmpi/' \
                'quiggeldy_setups_experimental.csv'
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


def save_nightly_calibration(filename: str = 'spiking_cocolist.pbin',
                             folder: Optional[str] = None):
    '''
    Save the nightly calibration to the given location.

    If the calibration is not available locally, it will be downloaded.

    :param filename: Name of the calibration to download. Typical names are
        'spiking_cocolist.pbin' and 'hagen_cocolist.pbin'.
    :param folder: Folder to save the calibration in. If not supplied the
        calibration is saved in the current folder.
    '''
    folder = Path() if folder is None else Path(folder)
    output_file = folder.joinpath(filename)
    if in_collaboratory():
        with hxcomm.ManagedConnection() as connection:
                identifier = connection.get_unique_identifier()

        download_url = "https://openproject.bioai.eu/data_calibration/" \
                       f"hicann-dls-sr-hx/{identifier}/stable/latest/" \
                       f"{filename}"
        urllib.request.urlretrieve(download_url, output_file)
    else:
        calib_path = pynn.helper.nightly_calib_path().parent
        path_to_calib =  calib_path.joinpath(filename)
        shutil.copy(path_to_calib, output_file)
