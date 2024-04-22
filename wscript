import os
from waflib.extras.test_base import summary
from waflib.extras.symwaf2ic import get_toplevel_path


def depends(dep):
    dep('haldls')
    dep("pynn-brainscales")
    dep("hxtorch")
    dep("libnux")
    dep('code-format')


def options(opt):
    opt.load("test_base")
    opt.load("shelltest")
    opt.load('pylint')
    opt.load('pycodestyle')


def configure(conf):
    conf.load("test_base")
    conf.load("shelltest")
    conf.load('pylint')
    conf.load('pycodestyle')


def build(bld):
    bld.env.BBS_HARDWARE_AVAILABLE = "SLURM_HWDB_YAML" in os.environ

    srcdir = bld.path.find_dir('.').get_src()
    blddir = bld.path.find_dir('.').get_bld()
    testdir = blddir.find_or_declare('test')
    sphinxbuild = "python -m sphinx"

    # Code style
    bld(name='much-demos-such-wow-static-code-analysis',
        features='use py pylint pycodestyle',
        source=bld.path.ant_glob('**/*.py'),
        use="pynn_brainscales2 dlens_vx_v3",
        pylint_config=os.path.join(get_toplevel_path(), "code-format", "pylintrc"),
        pycodestyle_config=os.path.join(get_toplevel_path(), "code-format", "pycodestyle"),
        test_timeout=60
    )

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
        test_timeout=1000,
        skip_run=not bld.env.BBS_HARDWARE_AVAILABLE
    )

    bld.add_post_fun(summary)
