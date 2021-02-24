from waflib.extras.test_base import summary


def depends(dep):
    dep("pynn-brainscales")


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

    bld(name="doc-much-demos-such-wow_shelltests",
        tests=bld.path.ant_glob("tests/shell/**/*"),
        features="use shelltest",
        use="doc-much-demos-such-wow-jupyter pynn_brainscales2",
        test_environ=dict(BLD_DIR=str(blddir)))

    bld.add_post_fun(summary)
