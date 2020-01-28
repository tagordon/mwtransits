import numpy as np
import exoplanet as xo
import matplotlib.pyplot as pl
import theano
import theano.tensor as tt

log_s0 = 0
log_w0 = 0
log_Q = 0
t = np.linspace(0, 1, 100)
kernel = xo.gp.terms.SHOTerm(log_S0=log_s0, log_w0=log_w0, log_Q=log_Q)
gp = xo.gp.GP(kernel=kernel, diag=np.ones(len(t)), x=t)

def benchmark(N, M, J, maxtime=1.0, trails=3, batch=10):
    t = np.linspace(0, 1, N)
    y = np.random.randn(N)
    kernel = xo.gp.terms.RealTerm(a=1, c=1)
    for i in range(J-1):
        kernel += xo.gp.terms.RealTerm(a=1, c=1)
    a = np.linspace(1, 2, M)
    Q = a[:, None]*a[None, :]
    gp = xo.gp.GP(kernel=kernel, diag=np.ones(N), x=t, Q=Q)
    
    y = tt.dmatrix()
    f = theano.function([y], gp.log_likelihood(y))
    timer = timeit.Timer("f(y)", setup="from __main__ import f, y")
    
    total = 0
    k = 0
    while total < maxtime:
        total += min(timer.repeat(trials, batch))
        k += 1
    return total / (batch*k)
    
# benchmarks for M as a function of N
M = np.linspace(1, 5, 5)
N = np.linspace(100, 10000, 10)

# benchmarks for M as a function of J 
M = np.linspace(1, 5, 5)
J = np.linspace(1, 100, 10)

# benchmarks for J as a function of N
J = np.linspace(1, 10, 5)
N = np.linspace(100, 10000, 10)

# benchmarks for J as a function of M
J = np.linspace(1, 10, 5)
M = np.linspace(1, 10, 10)

# benchmarks for N as a function of M
N = np.linspace(2000, 10000, 2000)
M = np.linspace(1, 10, 10)

# benchmarks for N as a function of J
N = np.linspace(2000, 10000, 2000)
J = np.linspace(1, 100, 10)
