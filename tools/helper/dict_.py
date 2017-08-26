from pandas import Series

from .file import establish_path


def merge_dicts_with_function(dict0, dict1, function):
    """
    Merge dict0 and dict1, apply function to values keyed by the same key.
    Arguments:
        dict0 (dict);
        dict1 (dict):
        function (callable):
    Returns:
        dict: merged dict
    """

    dict_ = {}

    for k in dict0.keys() | dict1.keys():

        if k in dict0 and k in dict1:
            dict_[k] = function(dict0.get(k), dict1.get(k))

        elif k in dict0:
            dict_[k] = dict0.get(k)

        else:
            dict_[k] = dict1.get(k)

    return dict_


def write_dict(dict_, file_path, key_name, value_name):
    """
    Write dict_ as .txt file with key and val columns.
    Arguments:
        dict_ (dict):
        file_path (str):
        key_name (str):
        value_name (str):
    Returns:
        None
    """

    s = Series(dict_, name=value_name)
    s.index.name = key_name

    if not file_path.endswith('.dict.txt'):
        file_path += '.dict.txt'

    establish_path(file_path)

    s.to_csv(file_path, sep='\t')
