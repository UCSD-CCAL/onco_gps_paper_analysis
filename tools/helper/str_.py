def title(str_):
    """
    Title str_.
    :param str_: str;
    :return: str;
    """

    # Remember indices of original uppercase letters
    uppers = []

    start = end = None
    is_upper = False
    for i, c in enumerate(str_):

        if c.isupper():
            if is_upper:  # Increment end
                end += 1

            else:  # Initialize start and end
                is_upper = True
                start = i
                end = start + 1

        else:
            if is_upper:  # Save start and end indices pair and reset
                is_upper = False
                uppers.append((start, end))
                start = end = None

    else:
        if start:  # Save the last start and end indices pair
            uppers.append((start, end))

    # Title
    str_ = str_.title().replace('_', ' ')

    # Upper all original uppercase letters
    for start, end in uppers:
        str_ = str_[:start] + str_[start:end].upper() + str_[end:]

    # Lower some words
    for lowercase in [
            'a',
            'an',
            'the',
            'and',
            'but',
            'or',
            'for',
            'nor',
            'on',
            'at',
            'to',
            'from',
            'of',
            'vs',
    ]:
        str_ = str_.replace(' ' + lowercase.title() + ' ',
                            ' ' + lowercase + ' ')

    return str_


def untitle(str_):
    """
    Untitle str_.
    :param str_: str;
    :return: str;
    """

    return str_.lower().replace(' ', '_').replace('-', '_')


def cast_builtins(str_):
    """
    Cast str_ in the order of int, float, bool, and str, returning the 1st
    successful casting.
    :param str_: str;
    :return: int, float, bool, or str; 1st successful casting
    """

    for type_ in (int, float):
        try:
            return type_(str_)
        except ValueError:
            pass

    if str_ == 'True':
        return True

    elif str_ == 'False':
        return False

    elif str_ == 'None':
        return None

    else:
        return str_


def split_ignoring_inside_quotes(str_, separator):
    """
    Split str_ by separator outside of quotes.
    :param str_: str;
    :param separator: str;
    :return: list; of str
    """

    list_ = []

    temp = ''
    for s in str_.split(separator):
        if '"' in s:
            if temp:
                temp += s
                list_.append(temp)
                temp = ''
            else:
                temp += s
        else:
            if temp:
                temp += s
            else:
                list_.append(s)

    return list_
