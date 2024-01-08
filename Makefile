# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -E
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = sphinxcontrib-jupyterminimal
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


# in order to have -W option for html (as does waf build flow, to
# take care of all warnings), html target has an explicit target
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"/html $(SPHINXOPTS) -W $(O)

# in order to be able to build in parallel easily
jupyter:
	@$(SPHINXBUILD) -M jupyter "$(SOURCEDIR)" "$(BUILDDIR)"/jupyter $(SPHINXOPTS) $(O)
latex:
	@$(SPHINXBUILD) -M latex "$(SOURCEDIR)" "$(BUILDDIR)"/latex $(SPHINXOPTS) $(O)
latexpdf:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)"/latex $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
