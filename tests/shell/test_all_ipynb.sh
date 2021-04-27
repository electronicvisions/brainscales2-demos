#!/bin/bash
set -euo pipefail

exec_test() {
    local testfile
    testfile="$1"
    cd "$(dirname "${testfile}")"
    ipython --colors=NoColor -c\
        "import IPython; IPython.get_ipython().safe_execfile_ipy('$(basename ${testfile})', raise_exceptions=True)"
}
export -f exec_test

find "${BLD_DIR}" -name "*.ipynb" -type f -print0 | xargs -0 -n 1 -I{} bash -c "exec_test '{}'"
