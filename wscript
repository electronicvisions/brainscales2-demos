import os
from waflib.extras.test_base import summary


def depends(dep):
    dep("pynn-brainscales")
    # These packages are not supported in EBRAINS release 1.0:
    #dep("hxtorch")
    #dep("libnux")


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
        rule=f'{sphinxbuild} -M jupyter {srcdir} {testdir} -D jupyter_drop_tests=0',
        always=True)

    # Build HTML
    bld(name='doc-much-demos-such-wow-html',
        rule=f'{sphinxbuild} -M html {srcdir} {blddir} -W',
        always=True)

    # Build target not supported in EBRAINS release 1.0:

    ## Build and install PPU code
    #bld.program(
    #    target='_static/tutorial/plasticity_kernel.bin',
    #    features='cxx',
    #    source='_static/tutorial/plasticity_kernel.cpp',
    #    use=['nux_vx_v2', 'nux_runtime_vx_v2'],
    #    env=bld.all_envs['nux_vx_v2'],
    #)

    #bld.install_files(
    #    testdir.find_or_declare('jupyter').find_or_declare('_static').find_or_declare('tutorial'),
    #    ['_static/tutorial/plasticity_kernel.bin'],
    #    name='install_plasticity_kernel_test',
    #    use="plasticity_kernel.bin"
    #)

    #bld.install_files(
    #    blddir.find_or_declare('jupyter').find_or_declare('_static').find_or_declare('tutorial'),
    #    ['_static/tutorial/plasticity_kernel.bin'],
    #    name='install_plasticity_kernel_deployment',
    #    use="plasticity_kernel.bin"
    #)

    # HW test
    bld(name="doc-much-demos-such-wow_shelltests",
        tests=bld.path.ant_glob("tests/shell/**/*"),
        features="use shelltest",
        #use="doc-much-demos-such-wow-jupyter-test pynn_brainscales2 hxtorch install_plasticity_kernel_test",
        use="doc-much-demos-such-wow-jupyter-test pynn_brainscales2",
        test_environ=dict(BLD_DIR=str(testdir)),
        test_timeout=600,
        skip_run=not bld.env.DLSvx_HARDWARE_AVAILABLE
    )

    bld.add_post_fun(summary)
