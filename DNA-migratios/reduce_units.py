from Params import Params


def reduce_x(x):
    return x / Params.L

def reduce_t(t):
    return t * Params.Omega

def reduce_units(x, t):
    return reduce_x(x), reduce_t(t)

def inverse_reduce_x(x):
    return x * Params.L

def inverse_reduce_t(t):
    return t / Params.Omega

def inverse_reduce_units(x, t):
    return inverse_reduce_x(x), inverse_reduce_t(t)


