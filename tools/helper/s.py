from pandas import DataFrame, Series

from .iterable import sanitize_nans
from .str_ import cast_builtins


def cast_builtins(s):
    """
    Cast s' objects in the order of int, float, bool, and str, returning the
    1st successful casting.
    :param s: Series; (n)
    :return: Series; (n)
    """

    list_ = sanitize_nans([cast_builtins(o) for o in s])

    try:
        return Series(list_, index=s.index, dtype=float)
    except TypeError as e:
        print(e)
    except ValueError as e:
        print(e)

    return Series(list_, index=s.index)


def make_membership_df(s):
    """
    Make a object-x-sample membership df (binary) from s.
    :param s: Series; (n_samples)
    :return: DataFrame; (n_unique_objects, n_samples)
    """

    # Make an empty DataFrame (n_unique_objects, n_samples)
    o_x_s = DataFrame(index=s.unique().sort_values(), columns=s.index)

    for o in o_x_s.index:
        o_x_s.ix[o, :] = (s == o).astype(int)

    return o_x_s
