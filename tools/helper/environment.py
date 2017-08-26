from os import environ

from pip import get_installed_distributions, main

from .str_ import cast_builtins
from .subprocess_ import run_command


def source_environment(file_path):
    """
    Source file path and update environment.
    :param file_path: str; file path to source
    :return: None
    """

    for line in run_command('./{}; env'.format(file_path)).stdout:

        i = line.find('=')
        k, v = line[:i].strip(), line[i + 1:].strip()

        environ[k] = v
        print('{} = {}'.format(k, v))


def install_libraries(libraries):
    """
    Install libraries that are not installed.
    :param libraries: iterable; of str, library names
    :return: None
    """

    # Get currently installed libraries
    libraries_installed = [lib.key for lib in get_installed_distributions()]

    # Install libraries not found in the currently installed libraries
    for lib in libraries:
        if lib not in libraries_installed:
            print('Installing {} ...'.format(lib))
            main(['install', lib])
        else:
            print('{} is already installed.'.format(lib))


def get_reference(obj, namespace):
    """
    If obj is in namespace, return its reference. Else cast it with built-in
    types. Unknown obj will be casted as str.
    :param obj: object; object
    :param namespace: dict; {ref: obj, ...}
    :return: int, float, bool, or str;
    """

    for r, o in namespace.items():
        if obj is o:  # obj is an existing obj
            return r

    # obj is a built-in type (or not in the namespace)
    return cast_builtins(obj)
