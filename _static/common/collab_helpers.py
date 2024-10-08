import os


def in_ebrains_collaboratory():
    lab_image_name = os.environ.get('LAB_IMAGE_NAME')
    if lab_image_name is None:
        return False

    return "ebrains" in lab_image_name.lower()


def check_kernel():
    expected_kernel = 'EBRAINS-experimental'
    actual_kernel = os.environ.get('LAB_KERNEL_NAME', None)
    if actual_kernel is None:
        raise RuntimeError(
            "Could not identify EBRAINS kernel (probably too old version). "
            f"Please select the appropriate kernel {expected_kernel}.")
    if actual_kernel != expected_kernel:
        raise RuntimeError(
            f"EBRAINS kernel mismatch. Expected: {expected_kernel} "
            f"Actual: {actual_kernel}. Please select the "
            "appropriate kernel.")
