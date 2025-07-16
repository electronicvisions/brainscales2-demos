import os
import urllib.request
from typing import Optional
from pathlib import Path
import shutil

import hashlib
import pandas as pd

from _static.common.collab_helpers import in_ebrains_collaboratory, \
    check_kernel

if in_ebrains_collaboratory():
    # check for EBRAINS kernel version prior to pynn_brainscales import
    check_kernel()

# pylint: disable=wrong-import-position
from dlens_vx_v3 import hxcomm
import pynn_brainscales.brainscales2 as pynn
# pylint: enable=wrong-import-position

logger = pynn.logger.get("demo_helpers")
pynn.logger.set_loglevel(logger, pynn.logger.LogLevel.INFO)


def is_experimental_kernel() -> bool:
    """
    Return whether we run in the experimental kernel.
    """
    return "experimental" in os.getenv("LAB_KERNEL_NAME")


def setup_hardware_client():
    if in_ebrains_collaboratory():
        setup_url = 'https://brainscales-r.kip.uni-heidelberg.de:7443/nmpi/' \
                    'quiggeldy_setups_experimental.csv'
        quiggeldy_setups = pd.read_csv(setup_url, dtype=str)

        os.environ['QUIGGELDY_ENABLED'] = '1'

        username = os.environ.get('JUPYTERHUB_USER')
        assert username is not None
        os.environ['QUIGGELDY_USER_NO_MUNGE'] = username

        index = int(hashlib.sha256(username.encode()).hexdigest(), 16) \
            % len(quiggeldy_setups)
        my_setup = quiggeldy_setups.iloc[index]
        os.environ['QUIGGELDY_IP'] = my_setup['Host']
        os.environ['QUIGGELDY_PORT'] = my_setup['Port']
    elif os.environ.get("QUIGGELDY_IP") is not None and\
            os.environ.get("QUIGGELDY_PORT") is not None:
        # quiggeldy port and IP manually set
        os.environ['QUIGGELDY_ENABLED'] = '1'
        os.environ['QUIGGELDY_USER_NO_MUNGE'] = os.environ.get("USER")
    elif os.environ.get("SLURM_FPGA_IPS") is not None:
        # on cluster with Slurm-managed resources
        return
    else:
        # not in collab, on cluster, or with manually configured
        # quiggeldy connection
        raise Exception("No proper hardware connection is possible check the "
                        "setup (detected to not be in ebrains, have quiggeldy "
                        "available or slurm in reach)")

    with hxcomm.ManagedConnection() as connection:
        identifier = connection.get_unique_identifier()

    logger.INFO(f'Connection to {identifier} established')


def get_nightly_calibration(filename='spiking_cocolist.pbin'):
    '''
    Get the nightly deployed calibration.

    If the code is executed in the EBrains in the collabatory the
    calibration is downloaded otherwise it is expected to be present
    locally.

    '''
    if in_ebrains_collaboratory():
        with hxcomm.ManagedConnection() as connection:
            identifier = connection.get_unique_identifier()

        # download calibration file
        version = "experimental" if is_experimental_kernel() else "stable"
        download_url = "https://openproject.bioai.eu/data_calibration/" \
                       f"hicann-dls-sr-hx/{identifier}/stable/" \
                       f"ebrains-{version}/{filename}"
        with urllib.request.urlopen(download_url) as response:
            contents = response.read()

        path_to_calib = filename
        with open(path_to_calib, 'wb') as f_handle:
            f_handle.write(contents)
    else:
        calib_path = pynn.helper.nightly_calib_path().parent
        path_to_calib = calib_path.joinpath(filename)

    chip = pynn.helper.chip_from_file(path_to_calib)

    return chip


def save_nightly_calibration(filename: str = 'spiking_cocolist.pbin',
                             source_folder: Optional[str] = None,
                             folder: Optional[str] = None):
    '''
    Save the nightly calibration to the given location.

    If the calibration is not available locally, it will be downloaded.

    :param filename: Name of the calibration to download. Typical names are
        'spiking_cocolist.pbin' and 'hagen_cocolist.pbin'.
    :param source_folder: Source folder to download the calibration from.
    :param folder: Folder to save the calibration in. If not supplied the
        calibration is saved in the current folder.
    '''

    folder = Path() if folder is None else Path(folder)
    output_file = folder.joinpath(filename)
    if source_folder is None:
        if in_ebrains_collaboratory():
            version = "experimental" if is_experimental_kernel() else "stable"
            source_folder = f"ebrains-{version}"
        else:
            source_folder = "latest"

    if in_ebrains_collaboratory():
        with hxcomm.ManagedConnection() as connection:
            identifier = connection.get_unique_identifier()

        download_url = \
            "https://openproject.bioai.eu/data_calibration/" \
            f"hicann-dls-sr-hx/{identifier}/stable/{source_folder}/" \
            f"{filename}"
        urllib.request.urlretrieve(download_url, output_file)
    else:
        calib_path = pynn.helper.nightly_calib_path().parent.parent
        path_to_calib = calib_path.joinpath(source_folder, filename)
        shutil.copy(path_to_calib, output_file)
