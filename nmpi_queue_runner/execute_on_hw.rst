.. code:: ipython3

    # Directory where we save the experiment results
    outputDir = os.path.expanduser("~")

    def evaluate_stable_release(kernel):
        """
        Checks if used kernel is the latest stable release
        """
        from datetime import datetime
        import re

        used_kernel_date = re.search("EBRAINS-(\d+.\d+)", kernel)
        if not used_kernel_date:
            raise RuntimeError('Unknown kernel used. Please either select '
                               '"EBRAINS_experimental_release" or the latest stable release')
        else:
            used_kernel_date = datetime.strptime(used_kernel_date.group(1), '%y.%m')

        available_kernels = !ls /opt/app-root/src/.local/share/jupyter/kernels
        kernel_dates = [re.search("spack_python_kernel_release_(\d{6})$", kernel) for kernel in available_kernels]
        latest_kernel_date = max([datetime.strptime(date.group(1), '%Y%m') for date in kernel_dates if date])
        return used_kernel_date == latest_kernel_date

    def execute_on_hardware(script_name):
        """
        Sends the provided script to the local cluster, where it is scheduled and executed on
        the neuromorphic chip. The result files are then loaded back into the collaboratory.
        Supported kernels are the latest stable relase and the experimental release.
        :param script_name: Name of the script which gets executed
        :returns: Job id of executed job
        """
        collab_id = repoInfo.nameInTheUrl

        # Evaluate used kernel
        kernel_env = os.getenv("LAB_KERNEL_NAME")
        if kernel_env is None:
            raise RuntimeError('Unknown kernel used. Please either select '
                               '"EBRAINS_experimental_release" or the latest stable release')
        if "experimental" in kernel_env:
            software_version = "experimental"
        elif evaluate_stable_release(kernel_env):
            software_version = "stable"
        else:
            raise RuntimeError('Selected kernel is not the latest stable release. Please either select '
                               '"EBRAINS_experimental_release" or the latest stable release')

        hw_config = {'SOFTWARE_VERSION': software_version}

        StartAt=time.time()
        # if connection broken, you need a new token (repeat the steps above)
        job = client.submit_job(source="~/"+script_name,
                              platform=nmpi.BRAINSCALES2,
                              collab_id=collab_id,
                              config=hw_config,
                              command="run.py",
                              wait=True)

        timeUsed = time.time() - StartAt
        job_id = job['id']
        print(str(job_id) + " time used: " + str(timeUsed))
        filenames = client.download_data(job, local_dir=os.path.expanduser("~"))
        print("All files: ",filenames)
        return job_id
