.. code-block:: ipython3
    :class: test, html-display-none

    # This env variable is not defined in our environment
    import os
    import getpass
    os.environ['JUPYTERHUB_USER']=getpass.getuser()

.. code:: ipython3

    import os
    os.environ['QUIGGELDY_USER_NO_MUNGE']=os.environ['JUPYTERHUB_USER']
    os.environ['QUIGGELDY_ENABLED']='1'
    os.environ['QUIGGELDY_IP']='129.206.127.39'
    os.environ['QUIGGELDY_PORT']='11717'

.. code-block:: ipython3
    :class: test, html-display-none

    # Do not use quiggeldy for tests
    os.environ['QUIGGELDY_ENABLED']='0'
