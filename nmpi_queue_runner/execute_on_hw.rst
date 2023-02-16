.. code:: ipython3

    from _static.common.collab_helpers import check_kernel

    # Check if compatible kernel version is used
    check_kernel()
    kernel_env = os.getenv("LAB_KERNEL_NAME")
    if "experimental" in kernel_env:
        software_version = "experimental"
    else:
        software_version = "stable"

    # Directory where we save the experiment results
    outputDir = os.path.expanduser("~")

    def execute_on_hardware(script_name):
        """
        Sends the provided script to the local cluster, where it is scheduled and executed on
        the neuromorphic chip. The result files are then loaded back into the collaboratory.
        :param script_name: Name of the script which gets executed
        :returns: Job id of executed job
        """
        collab_id = repoInfo.nameInTheUrl

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
