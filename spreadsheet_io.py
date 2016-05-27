import numpy as np
import pandas


def apply_and_append(file_path_or_data_frame, func, return_identifiers, argument_identifiers=None):
    """
    Reads in a spreadsheet / accepts a pandas data frame, interprets
    groups specified in argument_identifiers as arguments to a
    function func, and writes out the return values of func under the
    groups in return_identifiers.

    Arguments:
    ----------
        file_path_or_data_frame: str or pandas.DataFrame instance
        func: function handle
        return_identifiers: [str, str, ..., str]
        argument_identifiers: [str, str, ..., str] or None (default)

    Returns:
    --------
        none_or_data_frame: None or pandas.DataFrame

    """
    return

def _read(file_path):

    return data_frame

def _write(data_frame, file_path):
    return

def _apply(data_frame, func, argument_identifiers=None):
    return return_values

def _append(groups, values, data_frame):
    return data_frame

def _data_frame_to_dicts(data_frame):
    return dicts
