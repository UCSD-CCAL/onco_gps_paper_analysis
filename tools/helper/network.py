from os.path import abspath, join
from socket import AF_INET, SOCK_STREAM, socket

from requests import get


def download(url, directory_path=None):
    """
    Download url content.
    Arguments:
        url (str):
        directory_path (str): Directory path to download to
    Returns:
        str: File path to the downloaded content
    """

    r = get(url, stream=True)

    if not directory_path:
        directory_path = abspath('.')

    file_path = join(directory_path, url.split('/')[-1])

    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    return file_path


def get_open_port():
    """
    Get an open port.
    Arguments:
        None
    Returns:
        None
    """

    s = socket(AF_INET, SOCK_STREAM)

    s.bind(('', 0))

    port = s.getsockname()[1]

    s.close()

    return port
