import numpy
def is_nparray(a):
    if type(a) is not numpy.ndarray:
        return False

    return True
