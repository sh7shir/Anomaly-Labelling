import pandas as pd


def init_file_pandas(filename: str, columns: list):
    """
    A general purpose function for quickly initialising csv files using pandas.
    :param filename: The path leading to the file.
    :param columns: The desired column names.
    :return: No return value.
    """
    data = pd.DataFrame(columns=columns)
    data.to_csv(filename, index=False)
    return


def append_file_pandas(filename: str, new_row: list):
    """
    A general purpose function for quickly appending csv files using pandas.
    :param filename: The path leading to the file.
    :param new_row: The data to be inserted into the file.
    :return: No return value.
    """
    data = pd.read_csv(filename)
    update_index = len(data)
    data.loc[update_index] = new_row
    data.to_csv(filename, index=False)
    return
