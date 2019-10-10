import numpy as np
import pandas

"""
TODO: save out after each row (but actually this might not be a great idea... -> deferred)
"""

def apply_and_append(filepath_or_dataframe, func, arguments, returns):
    """
    Reads in a spreadsheet / accepts a pandas data frame, loops over
    rows, and interprets columns specified in 'arguments' as arguments
    to a function 'func', and writes out the return values of 'func'
    under the columns in 'returns'.

    The only key word (and thus column header) reserved is 'skip'.
    Rows for which 'skip' evaluates to True () are not processed.

    Arguments:
    ----------
        filepath_or_dataframe: str or pandas.DataFrame instance
            File path of data frame instance of a spreadsheet that specifies arguments for func.
        func: function handle
            Function to loop over while passing arguments specified in the spreadsheet.
        arguments: [str, str, ..., str]
            Columns corresponding to the arguments of func.
        returns: [str, str, ..., str]
            Columns corresponding to the return values of func.
            Provide an empty list if no return values should be saved,
            or if the function does not return anything.

    Returns:
    --------
        none_or_dataframe: None or pandas.DataFrame
            If a file path was supplied, results are saved in the same file
            and the functions returns nothing.
            If a data frame was supplied, the data frame is returned.

    """

    # get data
    if isinstance(filepath_or_dataframe, str):
        df = _read(filepath_or_dataframe)
    elif isinstance(filepath_or_dataframe, pandas.DataFrame):
        df = filepath_or_dataframe
    else:
        raise ValueError("filepath_or_dataframe neither a string nor a pandas.DataFrame object!" + \
                         "\ntype(filepath_or_dataframe) = {}".format(type(filepath_or_dataframe)))

    # trim empty rows
    df = df.dropna(how='all')

    # loop over rows
    total_rows = df.shape[0]
    for ii in range(total_rows):
        # test whether to skip row
        if 'skip' in df.columns:
            if df.loc[ii, 'skip']:
                if not np.isnan(df.loc[ii, 'skip']): # nans evaluate to True in python...
                    continue # ... to next row without processing current row
        # create dictionary
        kwargs = _row_to_dict(df, ii, arguments)
        # pass as key word arguments to function
        return_values = func(**kwargs)
        # save out to data frame
        for col, val in zip(returns, return_values):
            df.loc[ii, col] = val

    # save out / return data frame
    if isinstance(filepath_or_dataframe, str):
        return _write(df, filepath_or_dataframe)
    else:
        return df

def _read(file_path):
    # parse file path to determine extension
    extension = file_path.split('.')[-1]

    # read
    if extension in ('xls', 'xlsx'):
        df = pandas.read_excel(file_path)
    elif extension == 'csv':
        df = pandas.read_csv(file_path)
    else:
        raise ValueError('Spread sheet needs to be a csv or excel (.xls, .xlsx) file! Extension of supplied file path is {}'.format(extension))

    return df

def _write(df, file_path):
    # parse file path to determine extension
    extension = file_path.split('.')[-1]

    # write
    if extension in ('xls', 'xlsx'):
        df.to_excel(file_path, index=False)
    elif extension == 'csv':
        df.to_csv(file_path, index=False)
    else:
        raise NotImplementedError

    return

def _row_to_dict(df, row, keys):
    kv = []
    for key in keys:
        value = df.loc[row, key]
        # check for nan
        if isinstance(value, float): # need to check that type is float as np.isnan does not accept strings
            if np.isnan(value):
                continue # i.e. skip key, value pair
        kv.append((key, value))

    return dict(kv)
