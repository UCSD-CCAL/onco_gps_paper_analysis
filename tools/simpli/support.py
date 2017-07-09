from os import environ, listdir, mkdir
from os.path import abspath, isdir, isfile, islink, join, split
from sys import platform


def establish_filepath(filepath):
    """
    If the path up to the deepest directory in filepath doesn't exist, make the path up to the deepest directory.
    :param filepath: str; filepath
    :return: None
    """

    # prefix/suffix
    prefix, suffix = split(filepath)
    prefix = abspath(prefix)

    # Get missing directories
    missing_directories = []
    while not (isdir(prefix) or isfile(prefix) or
               islink(prefix)):  # prefix isn't file, directory, or link
        missing_directories.append(prefix)

        # Check prefix's prefix next
        prefix, suffix = split(prefix)

    # Make missing directories
    for d in reversed(missing_directories):
        mkdir(d)
        print('Created directory {}.'.format(d))


def list_only_dirs(directory_path):
    """

    :param directory_path: str; directory with all libraries
    :return: list; sorted list of directories in directory_path
    """

    dirs = []
    for f in listdir(directory_path):
        fp = join(directory_path, f)
        if isdir(fp):
            dirs.append(fp)
    dirs = sorted(dirs)

    return dirs


def get_home_dir():
    """

    :return: str; user-home directory
    """

    if 'linux' in platform or 'darwin' in platform:
        home_dir = environ['HOME']
    elif 'win' in platform:
        home_dir = environ['HOMEPATH']
    else:
        raise ValueError('Unknown platform {}.'.format(platform))

    return home_dir


def title_str(str_):
    """
    Title a str_.
    :param str_: str;
    :return: str;
    """

    # Remember indices of original uppercase letters
    uppers = []
    start = end = None
    is_upper = False
    for i, c in enumerate(str_):
        if c.isupper():
            # print('{} is UPPER.'.format(c))
            if is_upper:
                end += 1
            else:
                is_upper = True
                start = i
                end = start + 1
                # print('Start is {}.'.format(i))
        else:
            if is_upper:
                is_upper = False
                uppers.append((start, end))
                start = None
                end = start
    else:
        if start:
            uppers.append((start, end))

    # Title
    str_ = str_.title().replace('_', ' ')

    # Upper all original uppercase letters
    for start, end in uppers:
        str_ = str_[:start] + str_[start:end].upper() + str_[end:]

    # Lower some words
    for lowercase in [
            'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at',
            'to', 'from', 'of', 'vs', 'vs'
    ]:
        str_ = str_.replace(' ' + lowercase.title() + ' ',
                            ' ' + lowercase + ' ')

    return str_


def reset_encoding(str_):
    """

    :param str_: str;
    :return: str;
    """

    return str_.replace(u'\u201c', '"').replace(u'\u201d', '"')
