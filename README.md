# sphinxcontrib-jupyter.minimal

A Minimal Example of the Jupyter Sphinx Extension

This is a minimal example of a Sphinx project that is setup
to use the [Jupyter Extension](https://github.com/QuantEcon/sphinxcontrib-jupyter)

This requires you to install the extension:

    pip install sphinxcontrib-jupyter

> **Note:** This example currently only works using the latest [master](https://github.com/QuantEcon/sphinxcontrib-jupyter)
> and installing using `python setup.py install`. Awaiting for PyPI credentials to update releases. 

Once cloned you should be able to run:

    make jupyter

to generate a small set of demo jupyter notebooks.

Generated notebooks can be found in the `_build/jupyter/` folder.

## simple_notebook.rst

![simple_notebook.rst](_static/simple_notebook.png)

# Visionary Flow

Run in container (`singularity shell --app dls /containers/stable/latest`):

```shell
set -eou pipefail
virtualenv --system-site-packages venv
. venv/bin/activate
pip install sphinxcontrib-jupyter
make html
make jupyter
```

To look at the result file: `jupyter-notebook _build/jupyter/simple_notebook.ipynb`.
