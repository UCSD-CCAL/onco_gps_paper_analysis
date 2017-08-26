from os import environ, mkdir
from os.path import abspath, dirname, isdir, split, splitext
from sys import platform
from zipfile import ZipFile


def get_home_dir():
    """
    Get user-home directory.
    Arguments:
        None
    Returns:
        str: user-home directory
    """

    if platform in ('linux', 'darwin'):
        return environ['HOME']

    elif platform == 'win32':
        return environ['HOMEPATH']

    else:
        raise ValueError('Unknown platform: {}.'.format(platform))


def establish_path(path, path_type='file'):
    """
    Make directory paths up to the deepest directory in path.
    Arguments:
        path (str):
        path_type (str): 'file' | 'directory'
    Returns:
        None
    """

    if path.endswith('/'):
        raise ValueError('Path should not end with \'/\'')

    if path_type == 'directory':
        path += '/'

    directory_path, file_name = split(path)
    directory_path = abspath(directory_path)

    # List missing directory paths
    missing_directory_paths = []

    while not isdir(directory_path):

        missing_directory_paths.append(directory_path)

        # Check directory_path' directory_path
        directory_path, file_name = split(directory_path)

    # Make missing directories
    for d in reversed(missing_directory_paths):
        mkdir(d)
        print('Created directory {}/.'.format(d))


def mark_extension(file_path, mark):
    """
    Make 'path/to/file.mark.extension' from 'path/to/file.extension'.
    Arguments:
        file_path (str): 'path/to/file.extension'
        mark (str): 'mark'
    Returns:
        str: 'path/to/file.mark.extension'
    """

    root, extension = splitext(file_path)

    return '{}.{}.{}'.format(root, mark.strip('.'), extension)


def unzip(zip_file_path, directory_path=None):
    """
    Unzip zip_file_path in directory_path.
    Arguments:
        zip_file_path (str):
        directory_path (str):
    Returns:
        None
    """

    if not directory_path:
        directory_path = dirname(zip_file_path)

    print('Unzipping {} into {} ...'.format(zip_file_path, directory_path))
    with ZipFile(zip_file_path, 'r') as zf:
        zf.extractall(directory_path)


def read_file(file_path):
    """
    Read file_path content:
    Arguments:
        file_path (str):
    Returns:
        str: file_path content.
    """

    with open(file_path) as f:
        return f.read()
