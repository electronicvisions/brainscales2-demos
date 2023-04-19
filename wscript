import os
from waflib.extras.test_base import summary


def depends(dep):
    dep("pynn-brainscales")
    dep("hxtorch")
    dep("libnux")


def options(opt):
    opt.load("test_base")
    opt.load("shelltest")


def configure(conf):
    conf.load("test_base")
    conf.load("shelltest")


def build(bld):
    bld.env.DLSvx_HARDWARE_AVAILABLE = "cube" == os.environ.get("SLURM_JOB_PARTITION")

    srcdir = bld.path.find_dir('.').get_src()
    blddir = bld.path.find_dir('.').get_bld()
    testdir = blddir.find_or_declare('test')
    sphinxbuild = "python -m sphinx"

    # Build jupyter
    bld(name='doc-much-demos-such-wow-jupyter',
        rule=f'{sphinxbuild} -M jupyter {srcdir} {blddir}',
        always=True)

    bld(name='doc-much-demos-such-wow-jupyter-test',
        rule=f'{sphinxbuild} -M jupyter {srcdir} {testdir} -D jupyter_drop_tests=0 -t exclude_nmpi',
        always=True)

    # Build HTML
    bld(name='doc-much-demos-such-wow-html',
        rule=f'{sphinxbuild} -M html {srcdir} {blddir} -W',
        always=True)

    # HW test
    bld(name="doc-much-demos-such-wow_shelltests",
        tests=bld.path.ant_glob("tests/shell/**/*"),
        features="use shelltest",
        use="doc-much-demos-such-wow-jupyter-test pynn_brainscales2 hxtorch",
        test_environ=dict(BLD_DIR=str(testdir)),
        test_timeout=360,
        skip_run=not bld.env.DLSvx_HARDWARE_AVAILABLE
    )

    bld.add_post_fun(summary)
