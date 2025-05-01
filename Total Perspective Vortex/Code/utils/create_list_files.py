
def create_list_files(subjects, runs, root):
    """
    Create a list of files to be downloaded from PhysioNet.
    Parameters
    ----------
    subjects : list
        List of subjects to be downloaded.
    runs : list
        List of runs to be downloaded.
    Returns
    -------
    list
        List of files to be downloaded.
    """
    list = []
    if subjects is None:
        return None
    for subject in subjects:
      folder = "S" + f"{subject:03}" + "/"
      for run in runs:
        file1 = "S"+ f"{subject:03}" +"R" + f"{run:02}" + ".edf"
        if runs is not None:
            link = root + folder + file1
            list.append(link)
        else:
            link = root + folder
            list.append(link)
    return list