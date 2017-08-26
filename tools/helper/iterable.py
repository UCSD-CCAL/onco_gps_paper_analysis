from numpy import nan, where


def integize(iterable):
    """
    Integize iterable's objects.
    :param iterable: iterable; of objects;
    :return: list; of int
    """

    # Assign objects to ints
    o_to_i = {}
    for i, o in enumerate(sorted(set(iterable))):
        o_to_i[o] = i

    # Map objects to ints and return
    return [o_to_i[o] for o in iterable]


def group(iterable, n, keep_leftover_group=False):
    """
    Make a list of groups, each containing n objects from iterable in order.
    :param iterable: iterable; whose objects are grouped
    :param n: int; number of objects in a group
    :param keep_leftover_group: bool; keep the partialy-leftover group or not
    :return: list; of groups, each containing n objects form iterable in order
    """

    list_ = []

    group = []
    for o in iterable:

        group.append(o)

        if len(group) == n:
            list_.append(group)
            group = []

    # Handle leftover
    if len(group) != 0 and (len(group) == n or keep_leftover_group):
        list_.append(group)

    return list_


def flatten_nested(iterable, iterable_types=(list, tuple)):
    """
    Flatten an arbitrarily-deep nested iterable by depth-first flattening.
    :param iterable: of iterables
    :param iterable_types: iterable types to be flattened
    :return: list; a flattened list
    """

    list_ = list(iterable)

    i = 0
    while i < len(list_):

        while isinstance(list_[i], iterable_types):

            if not list_[i]:  # Skip empty iterable
                list_.pop(i)
                i -= 1
                break

            else:  # Deepen
                list_[i:i + 1] = list_[i]

        i += 1

    return list_


def get_uniques_in_order(iterable):
    """
    Get unique objects in the order of their appearance in iterable.
    :param iterable: iterable; of objects
    :return: list; of unique objects ordered by their appearances in iterable
    """

    list_ = []

    for o in iterable:

        if o not in list_:
            list_.append(o)

    return list_


def sanitize_nans(iterable,
                  nans=('--', 'unknown', 'n/a', 'N/A', 'na', 'NA', 'nan',
                        'NaN', 'NAN')):
    """
    Convert nan-equivalent objects into np.nan.
    :param iterable: iterable; of objects
    :param nans: iterable; of objects
    :return: list;
    """

    return [where(v in nans, nan, v) for v in iterable]
