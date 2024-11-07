import numpy as np

def MAEDistance(x, y):
    x = x.asnumpy()  
    y = y.asnumpy()
    return np.mean(np.abs(x - y))

def ChebyshevDistance(x, y):
    x = x.asnumpy()  
    y = y.asnumpy()
    return np.max(np.abs(x - y))
