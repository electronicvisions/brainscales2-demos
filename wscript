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
    srcdir = bld.path.find_dir('.').get_src()
    blddir = bld.path.find_dir('.').get_bld()
    sphinxbuild = "python -m sphinx"
    bld(name='doc-much-demos-such-wow-jupyter', rule=f'{sphinxbuild} -M jupyter {srcdir} {blddir}', always=True)
    bld(name='doc-much-demos-such-wow-html', rule=f'{sphinxbuild} -M html {srcdir} {blddir}', always=True)

    bld.program(
        target = 'plasticity_kernel.bin',
        features = 'cxx',
        source = 'plasticity_kernel.cpp',
        use = ['nux_vx_v2', 'nux_runtime_vx_v2'],
        env = bld.all_envs['nux_vx_v2'],
    )

    bld(name="doc-much-demos-such-wow_shelltests",
        tests=bld.path.ant_glob("tests/shell/**/*"),
        features="use shelltest",
        use="doc-much-demos-such-wow-jupyter pynn_brainscales2 hxtorch plasticity_kernel.bin",
        test_environ=dict(BLD_DIR=str(blddir)),
        test_timeout=300,
    )

    bld.add_post_fun(summary)
