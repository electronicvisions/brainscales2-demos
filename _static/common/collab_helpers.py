import os

def in_collaboratory():
    return os.environ.get('JUPYTERHUB_USER') is not None

def check_kernel():
    expected_kernel = 'EBRAINS-experimental'
    actual_kernel = os.environ.get('LAB_KERNEL_NAME', None)
    if actual_kernel is None:
        raise RuntimeError(
            "Could not identify EBRAINS kernel (probably too old version). " +
            f"Please select the appropriate kernel {expected_kernel}.")
    elif actual_kernel != expected_kernel:
        raise RuntimeError(
            f"EBRAINS kernel mismatch. Expected: {expected_kernel} " +
            f"Actual: {actual_kernel}. Please select the " +
            "appropriate kernel.")
