from subprocess import PIPE, run


def run_command(command):
    """
    Execute command.
    :param command: str; command
    :return: str; command output
    """

    print('command: {}'.format(command))

    return run(command,
               shell=True,
               check=True,
               stdout=PIPE,
               universal_newlines=True)
