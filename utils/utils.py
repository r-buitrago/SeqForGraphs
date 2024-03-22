import operator
from functools import reduce
from timeit import default_timer as timer

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

def time_it(f):
    def f_(self, *args, **kwargs):
        start = timer()
        result = f(self, *args, **kwargs)
        end = timer()
        try:
            print(f"{f.__class__.__name__} took {end-start} seconds")
        except:
            print(f"{f.__name__} took {end-start} seconds")
        return result
    return f_