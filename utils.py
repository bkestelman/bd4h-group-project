import time
import functools


def timeit(func):
    # https://stackoverflow.com/questions/5478351/python-time-measure-function
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {:.2f} seconds'.format(func.__name__, elapsedTime))
        return result
    return newfunc