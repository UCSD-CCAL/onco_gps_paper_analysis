from multiprocessing.pool import Pool

from numpy.random import seed


def multiprocess(function, args, n_jobs, random_seed=None):
    """
    Call function with args across n_jobs processes (n_jobs doesn't have
    to be the length of list_of_args).
    :param function: callable;
    :param args: iterable; of args
    :param n_jobs: int; 0<
    :param random_seed: int | array;
    :return: list;
    """

    if random_seed is not None:
        # Each process initializes with the current jobs' randomness (random
        # state & random state index). Any changes to these jobs' randomnesses
        # won't update the current process' randomness.
        seed(random_seed)

    with Pool(n_jobs) as p:
        return p.map(function, args)
