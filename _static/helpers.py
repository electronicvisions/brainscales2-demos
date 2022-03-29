import os
import urllib.request

import hashlib
import pandas as pd

from dlens_vx_v2 import hxcomm
import pynn_brainscales.brainscales2 as pynn

def in_collaboratory():
    return os.environ.get('JUPYTERHUB_USER') is not None

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
