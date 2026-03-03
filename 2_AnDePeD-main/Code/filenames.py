import glob


def get_files(datasets: list, directory: str, file_format: str = '*', num_levels: int = 1):
    """
    Updates the list "datasets" with the list of files in the "directory" with the appropriate "format".
    Takes into account the number of directory layers supplied by the user.
    :param datasets: A list to put file paths into.
    :param directory: Location of files.
    :param file_format: File format, e.g. "csv".
    :param num_levels: Number of directory levels until the desired files are reached.
    :return: No return value.
    """
    if num_levels < 1:
        raise ValueError('num_levels has to be at least one.')
    datasets.clear()
    files = glob.glob(directory + (num_levels - 1) * '/**' + '/*.' + file_format, recursive=True)
    for file in files:
        datasets.append(file)
    return


def get_files_onelayer_andeped(folder: str, separator: str = '\\'):
    """
    Searches for AnDePeD test result files in the "folder" and returns their unique IDs (datetimes).
    :param folder: Directory to find AnDePeD test results in.
    :param separator: Character separator between folder levels (usually '\\' or '/').
    :return: Returns the list of datetimes (yyyymmddhhmmss).
    """
    tmp_files = glob.glob(folder + '/*_hyperparams.csv', recursive=True)
    datetimes = list()
    for file in tmp_files:
        datetimes.append(file.split(separator)[-1].split('_')[0])
    return datetimes


def separate_names(filename: str, separator: str = '\\', remove_csv: bool = True):
    """
    Separates a file path into separate strings based on directory levels.
    Use: a, b, c = separate_names('aaa/bbb/ccc.csv', '/'); result: a='aaa', b='bbb', c='ccc'.
    :param filename: File path.
    :param separator: Character separator between folder levels (usually '\\' or '/').
    :param remove_csv: Boolean that signals whether to remove '.csv' from the end of the last element.
    :return: Returns an array with each level's name.
    """
    names = filename.split(separator)
    if remove_csv:
        names[-1] = names[-1][:-4]
    return names


def replace_names(filename: str, to_replace: str, replace_by: str):
    """
    Replaces a string in "filename" by another string.
    :param filename: The file path that needs replacing.
    :param to_replace: Search for this...
    :param replace_by: ...and replace it by this.
    :return: The new file path.
    """
    return filename.replace(to_replace, replace_by)
