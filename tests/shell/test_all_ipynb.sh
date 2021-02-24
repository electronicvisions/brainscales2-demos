#!/bin/bash
set -euo pipefail

find "${BLD_DIR}" -name "*.ipynb" -type f -print0 | xargs -0 -I"{}" ipython --colors=NoColor -c\
  "import IPython; IPython.get_ipython().safe_execfile_ipy('{}', raise_exceptions=True)"
