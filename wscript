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
    hopts = opt.add_option_group('demos options')
    hopts.add_withoption('solution', default=False,
                         help = 'Build sphinx with "Solution" tag')
    hopts.add_withoption('latex', default=True, help = 'Build PDF.')


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
    # Add Tags to sphinx build (e.g. for solution)
    tags = ""
    if bld.options.with_solution:
        tags += "-t Solution"

    # Build jupyter
    bld(name='doc-much-demos-such-wow-jupyter',
        rule=f'{sphinxbuild} -M jupyter {srcdir} {blddir}/jupyter -E {tags}',
        always=True)

    bld(name='doc-much-demos-such-wow-jupyter-test',
        rule=(f'{sphinxbuild} -M jupyter {srcdir} {testdir} '
              + '-D jupyter_drop_tests=0 -t exclude_nmpi -t Solution'),
        always=True)

    # Build HTML
    bld(name='doc-much-demos-such-wow-html',
        rule=f'{sphinxbuild} -M html {srcdir} {blddir}/html -E {tags} -W',
        always=True)

    # Build PDF
    if bld.options.with_latex:
        bld(name='doc-much-demos-such-wow-pdf',
            rule=f'{sphinxbuild} -M latexpdf {srcdir} {blddir}/latex -E {tags}',
            always=False)


    # HW test
    bld(
        target='doc-much-demos-such-wow_ipynb_executability_tests',
        tests=bld.path.ant_glob('tests/py/*.py'),
        features='use pytest',
        use=['doc-much-demos-such-wow-jupyter-test', 'pynn_brainscales2', 'hxtorch'],
        install_path='${PREFIX}/bin/tests/py',
        test_environ=dict(BLD_DIR=str(testdir)),
        test_timeout=3600,
        skip_run=not bld.env.BBS_HARDWARE_AVAILABLE,
    )

    bld.add_post_fun(summary)
