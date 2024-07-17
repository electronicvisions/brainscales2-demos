"""
Test all ipynb whether they execute without errors
"""
import glob
import os
import os.path as osp
import subprocess
import unittest


class TestIpynb(unittest.TestCase):

    @classmethod
    def generate_cases(cls, path):
        """
        Augment this class by test cases for trying to execute all ipynb.
        """
        file_list = glob.glob(path, recursive=True)

        if len(file_list) == 0:
            raise IOError(f"Found no ipynb in the given path {path}")

        for notebook in file_list:
            def generate_test(filename: str):
                def test_func(_: TestIpynb):
                    subprocess.run(
                        [
                            'ipython',
                            '--colors=NoColor',
                            '-c',
                            "import IPython;"
                            "IPython.get_ipython().safe_execfile_ipy('"
                            f"{osp.basename(filename)}', "
                            "raise_exceptions=True)"
                        ],
                        cwd=osp.dirname(filename),
                        check=True,
                    )

                return test_func

            test_method = generate_test(notebook)
            test_method.__name__ = f"test_execute_{osp.basename(notebook)}"

            if hasattr(cls, test_method.__name__):
                raise RuntimeError(
                    f"notebook {notebook} would lead to test method with same "
                    f"name {test_method.__name__}"
                )
            setattr(cls, test_method.__name__, test_method)


TestIpynb.generate_cases(
    f"{os.environ['BLD_DIR']}/**/*.ipynb"
)


if __name__ == "__main__":
    unittest.main()
