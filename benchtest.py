import numpy as np
import exoplanet as xo
import matplotlib.pyplot as pl
import theano
import theano.tensor as tt
import itertools
import timeit

red = '#FE4365'
blue = '#00A9FF'
yellow = '#ECA25C'
green = '#3F9778'
darkblue = '#005D7F'
colors = [red, blue, yellow, green, darkblue]

pl.rc('xtick', labelsize=20)
pl.rc('ytick', labelsize=20)
pl.rc('axes', labelsize=25)
pl.rc('axes', titlesize=30)
pl.rc('legend', fontsize=20)
pl.rc('lines', linewidth=4)
pl.rc('lines', markersize=15)
pl.rc('lines', markeredgewidth=1.5)

log_s0 = 0
log_w0 = 0
log_Q = 0
t = np.linspace(0, 1, 100)
kernel = xo.gp.terms.SHOTerm(log_S0=log_s0, log_w0=log_w0, log_Q=log_Q)
gp = xo.gp.GP(kernel=kernel, diag=np.ones(len(t)), x=t)

def benchmark(N, M, J, maxtime=10.0, trials=3, batch=10):
    global f
    global y
    t = np.linspace(0, 1, N)
    y = np.random.randn(N)
    k = 2
    kernel = xo.gp.terms.SHOTerm(log_S0=log_s0, log_w0=log_w0, log_Q=log_Q)
    for i in range(np.int(J-1)):
        k += 2
        kernel += xo.gp.terms.SHOTerm(log_S0=log_s0, log_w0=log_w0, log_Q=log_Q)
    a = np.linspace(1, 2, M)
    Q = a[:, None]*a[None, :]
    print("given to gp: ", 2*M*J, "should be equal to: ", k*M)
    gp = xo.gp.GP(kernel=kernel, diag=np.ones((M, N)), x=t, Q=Q, J=np.int(2*M*J))
    
    z = tt.dmatrix()
    f = theano.function([z], gp.log_likelihood(z))
    y = np.random.randn(N*M)
    timer = timeit.Timer("f(y[None, :])", setup="from __main__ import f, y")
    
    total = 0
    k = 0
    print(f(y[None, :]))
    #while total < maxtime:
    #    total += min(timer.repeat(trials, batch))
    #    k += 1
    #return total / (batch*k)

# benchmarks for J as a function of N
J = np.arange(2, 12, 2)
N = np.arange(1000, 11000, 1000)
M = 2

res = np.zeros((len(J), len(N)))
for i, j in itertools.product(range(len(J)), range(len(N))):
    benchmark(N[j], M, J[i])
np.savetxt("data/benchmarks_JN.txt", res)


# benchmarks for J as a function of M
J = np.arange(2, 12, 2)
M = np.arange(1, 11, 1)
N = 100

res = np.zeros((len(J), len(M)))
for i, j in itertools.product(range(len(J)), range(len(M))):
    benchmark(N, M[j], J[i])
np.savetxt("data/benchmarks_JM.txt", res)


# benchmarks for N as a function of M
N = np.arange(2000, 10000, 2000)
M = np.arange(1, 11, 1)
J = 1

pl.figure(figsize=(10, 7))
res = np.zeros((len(N), len(M)))
for i, j in itertools.product(range(len(N)), range(len(M))):
    benchmark(N[i], M[j], J)
np.savetxt("data/benchmarks_NM.txt", res)
    
# benchmarks for N as a function of J
N = np.arange(2000, 10000, 2000)
J = np.arange(10, 110, 10)
M = 2

res = np.zeros((len(N), len(J)))
for i, j in itertools.product(range(len(N)), range(len(J))):
    benchmark(N[i], M, J[j])
np.savetxt("data/benchmarks_NJ.txt", res)