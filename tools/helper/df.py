from numpy import array
from pandas import concat

RANDOM_SEED = 20121020


def drop_slices(df, axis, only_obj=None, max_n_unique_objects=None):
    """
    Drop slices.
    :param df: DataFrame
    :param axis: int; 0 | 1
    :param only_obj: object
    :param max_n_unique_objects: int; 0 <
    :return: DataFrame
    """

    if only_obj is None and max_n_unique_objects is None:
        raise ValueError('Provide either only_obj or max_n_unique_objects.')

    # Select slices to be dropped
    dropped = array([False] * df.shape[[1, 0][axis]])

    if only_obj is not None:
        s = 'only {}'.format(only_obj)
        dropped |= (df == only_obj).all(axis=axis)

    if 0 < max_n_unique_objects:
        s = 'at most {} unique object(s)'.format(max_n_unique_objects)
        dropped |= df.apply(
            lambda s: s.unique().size <= max_n_unique_objects, axis=axis)

    # Drop
    if dropped.any():
        print('Dropping {} axis-{} slices containing {} ...'.format(
            dropped.sum(), axis, s))
        print('******** Dropped slices ********')
        print(df.index[dropped].tolist())
        print('********************************')

        if axis == 0:
            return df.ix[:, ~dropped]

        elif axis == 1:
            return df.ix[~dropped, :]

    else:
        return df.copy()


def split_df(df, n_split, axis=0):
    """
    Split df into n_split blocks.
    :param df: DataFrame
    :param n_split: int; 0 < n_split <= n_rows
    :param axis: int; 0 | 1
    :return: list; of DataFrame
    """

    if df.shape[axis] < n_split:
        raise ValueError('Number of slices ({}) < n_split ({})'.format(
            df.shape[axis], n_split))
    elif n_split <= 0:
        raise ValueError('n_split ({}) <= 0'.format(n_split))

    n = df.shape[axis] // n_split

    list_ = []

    for i in range(n_split):
        start_i = i * n
        end_i = (i + 1) * n

        if axis:
            list_.append(df.iloc[:, start_i:end_i])
        else:
            list_.append(df.iloc[start_i:end_i, :])

    # Get leftovers (if any)
    i = n * n_split
    if i < df.shape[axis]:

        if axis:
            list_.append(df.iloc[:, i:])
        else:
            list_.append(df.iloc[i:, :])

    return list_


def split_str_slice(df, index, splitter, axis=0):
    """
    Split str slice with splitter and replace this slice with 2 new slices made
        with the splits (copying rest of the objects).
    :param df: DataFrame
    :param index: str; index of the str slice
    :param splitter: str
    :param ax: int; 0 | 1
    :return: DataFrame; [n_rows (+1 if ax==1), n_cols (+1 if ax==0)]
    """

    new_slices = []

    if axis == 0:  # Split columns
        df = df.T

    for s_i, s in df.iterrows():

        str_ = s.ix[index]

        # Split str and make 2 new slices with the split str
        for str_split in str_.split(splitter):
            new_slices.append(s.replace(str_, str_split))

    # Concatenate the new slices and return the DataFrame
    if axis == 0:
        return concat(new_slices, axis=1)
    elif axis == 1:
        return concat(new_slices, axis=1).T


def get_top_and_bottom_indices(df, column, threshold, max_n=None):
    """
    Get the indices for the top & bottom rows in respect to column.
    :param df: DataFrame
    :param column: str; column name
    :param threshold: number; quantile if < 1; ranking number if 1 <=
    :param max_n: int; maximum number of rows
    :return: Index; indices for the top and bottom rows in respect to column
    """

    column = df[column]

    if threshold < 1:

        i = column.index[(column.quantile(threshold) <= column) | (
            column <= column.quantile(1 - threshold))]

        if max_n and max_n < i.size:  # Limit threshold
            threshold = max_n // 2

    if 1 <= threshold:

        if 2 * threshold <= column.size:
            rank = column.rank(method='dense')
            i = column.index[(rank <= threshold) | ((rank.max() - threshold) <
                                                    rank)]
        else:
            i = column.index

    return i
