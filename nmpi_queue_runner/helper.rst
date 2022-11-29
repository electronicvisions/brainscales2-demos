.. code:: ipython3

    # Define helpers to store and extract required information of the currently used collab
    class RepositoryInformation:
        def __init__(self, repository, repoInfo):
            self.repository = repository
            self.nameInTheUrl = repoInfo["name"]
            self.repoInfo = repoInfo

        def toString(self):
            return "nameInTheUrl=" + self.nameInTheUrl + ", full name=" + \
                self.repository.name + ", id=" + self.repository.id


    def findRepositoryInfoFromDriveDirectoryPath(homePath):
        # Remove directory structure and subsequent folder names to extract the name of the collab
        name = homePath.replace("/mnt/user/shared/", "")
        if name.find("/") > -1:
            name = name[:name.find("/")]
        bearer_token = clb_oauth.get_token()
        ebrains_drive_client = ebrains_drive.connect(token=bearer_token)
        repo_by_title = ebrains_drive_client.repos.get_repos_by_filter("name", name)
        if len(repo_by_title) != 1:
            raise Exception("The repository for the collab name", name,
                            "can not be found.")

        # With the repo_by_title we can get the drive ID
        driveID = repo_by_title[0].id

        # and then we can use the driveID to look up the collab
        url = "https://wiki.ebrains.eu/rest/v1/collabs?driveId=" + driveID
        response = requests.get(
            url,
            headers={'Authorization': 'Bearer %s' % bearer_token})
        repo_info = response.json()
        return RepositoryInformation(repo_by_title[0], repo_info)


    # Generate HBP client used to communicate with the hardware and extract
    # collab information from current working directory using the previously
    # defined helpers
    client = nmpi.Client()
    dir =!pwd
    repoInfo = findRepositoryInfoFromDriveDirectoryPath(dir[0])

    # Optionally: Set 'checkForQuota' to True to check if the currently used
    # collab has an existing quota
    checkForQuota = False
    if checkForQuota:
        a = client.list_resource_requests(repoInfo.nameInTheUrl)
        anyAccepted = False
        if len(a) == 0:
            print("This collab does not have any quota entry yet. Request a test quota. "
                  "This request will need to be reviewd and granted by an admin.")
            client.create_resource_request(
                title="Test quota request for " + repoInfo.nameInTheUrl,
                collab_id=repoInfo.nameInTheUrl,
                abstract="Test quota request",
                submit=True)
        else:
            for entry in a:
                if entry["status"] == "accepted":
                    print("An accepted quota request exists")
                    anyAccepted = True
            if not anyAccepted:
                print("A quota request is present, but it has not yet been granted.")
        if not anyAccepted:
            raise Exception(
                "This collab does not yet have an accepted quota entry. "
                "Therefore submitting jobs will not yet work.")
