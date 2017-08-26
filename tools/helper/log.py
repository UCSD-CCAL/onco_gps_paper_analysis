from datetime import datetime


def get_now(only_time=False):
    """
    Get the current datetime.
    Arguments:
        only_time (bool): Whether to show only time
    Returns:
        str: year-month-date hour-minute-second or hour-minute-second
    """

    if only_time:
        formatter = '%H:%M:%S'
    else:
        formatter = '%Y-%m-%d %H:%M:%S'

    return datetime.now().strftime(formatter)
