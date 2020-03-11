
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
    kernel = xo.gp.terms.SHOTerm(log_S0=log_s0, log_w0=log_w0, log_Q=log_Q)
    for i in range(np.int(J-1)):
        kernel += xo.gp.terms.SHOTerm(log_S0=log_s0, log_w0=log_w0, log_Q=log_Q)
    a = np.linspace(1, 2, M)
    Q = a[:, None]*a[None, :]
    gp = xo.gp.GP(kernel=kernel, diag=np.ones((M, N)), x=t, Q=Q, J=np.int(2*M*J))
    
    z = tt.dmatrix()
    f = theano.function([z], gp.log_likelihood(z))

    y = np.random.randn(N*M)
    timer = timeit.Timer("f(y[None, :])", setup="from __main__ import f, y")
    
    total = 0
    k = 0
    while total < maxtime:
        total += min(timer.repeat(trials, batch))
        k += 1
    return total / (batch*k)
    
# benchmarks for M as a function of N
#M = np.arange(1, 6, 1)
#N = np.arange(1000, 11000, 1000)
#J = 1
#
#res = np.zeros((len(M), len(N)))
#for i, j in itertools.product(range(len(M)), range(len(N))):
#    res[i, j] = benchmark(N[j], M[i], J)
#np.savetxt("data/benchmarks_MN.txt", res)

#pl.figure(figsize=(10, 7))
#for i in range(len(M)):
#    pl.loglog(N, res[i], '.-', color=colors[i%len(colors)], 
#              markeredgecolor='k', label="M={0}".format(M[i]))
#pl.legend()
#pl.xlabel("N (size of first dimension)")
#pl.ylabel("time (seconds)")
#pl.savefig("plots/benchmarks_MN.pdf")

# benchmarks for M as a function of J 
#M = np.arange(1, 6, 1)
#J = np.arange(10, 110, 10)
#N = 100

#res = np.zeros((len(M), len(J)))
#for i, j in itertools.product(range(len(M)), range(len(J))):
#    res[i, j] = benchmark(N, M[i], J[j])
#np.savetxt("data/benchmarks_MJ.txt", res)

#pl.figure(figsize=(10, 7))
#for i in range(len(M)):
#    pl.loglog(J, res[i], '.-', color=colors[i%len(colors)], 
#              markeredgecolor='k', label="M={0}".format(M[i]))
#pl.legend()
#pl.xlabel("J (number of terms in kernel)")
#pl.ylabel("time (seconds)")
#pl.savefig("plots/benchmarks_MJ.pdf")

# benchmarks for J as a function of N
#J = np.arange(10, 60, 10)
#N = np.arange(1000, 11000, 1000)
#M = 2
#
#res = np.zeros((len(J), len(N)))
#for i, j in itertools.product(range(len(J)), range(len(N))):
#    res[i, j] = benchmark(N[j], M, J[i])
#np.savetxt("data/benchmarks_JN.txt", res)

#pl.figure(figsize=(10, 7))
#for i in range(len(J)):
#    pl.loglog(N, res[i], '.-', color=colors[i%len(colors)], 
#              markeredgecolor='k', label="J={0}".format(2*J[i]))
#pl.legend()
#pl.xlabel("N (number of points in first dimension)")
#pl.ylabel("time (seconds)")
#pl.savefig("plots/benchmarks_JN.pdf")

# benchmarks for J as a function of M
J = np.arange(10, 60, 10)
M = np.arange(20, 100, 20)
N = 100

res = np.zeros((len(J), len(M)))
for i, j in itertools.product(range(len(J)), range(len(M))):
    res[i, j] = benchmark(N, M[j], J[i])
np.savetxt("data/benchmarks_JM.txt", res)

pl.figure(figsize=(10, 7))
for i in range(len(J)):
    pl.loglog(M, res[i], '.-', color=colors[i%len(colors)], 
              markeredgecolor='k', label="J={0}".format(2*J[i]))
pl.legend()
pl.xlabel("M (size of second dimension)")
pl.ylabel("time (seconds)")
pl.savefig("plots/benchmarks_JM.pdf")

# benchmarks for N as a function of M
N = np.arange(2000, 10000, 2000)
M = np.arange(20, 100, 20)
J = 1

pl.figure(figsize=(10, 7))
res = np.zeros((len(N), len(M)))
for i, j in itertools.product(range(len(N)), range(len(M))):
    res[i, j] = benchmark(N[i], M[j], J)
np.savetxt("data/benchmarks_NM.txt", res)

pl.figure(figsize=(10, 7))
for i in range(len(N)):
    pl.loglog(M, res[i], '.-', color=colors[i%len(colors)], 
              markeredgecolor='k', label="N={0}".format(N[i]))
pl.legend()
pl.xlabel("M (size of second dimension)")
pl.ylabel("time (seconds)")
pl.savefig("plots/benchmarks_NM.pdf")
    
# benchmarks for N as a function of J
#N = np.arange(2000, 10000, 2000)
#J = np.arange(10, 110, 10)
#M = 2

#res = np.zeros((len(N), len(J)))
#for i, j in itertools.product(range(len(N)), range(len(J))):
#    res[i, j] = benchmark(N[i], M, J[j])
#np.savetxt("data/benchmarks_NJ.txt", res)

#pl.figure(figsize=(10, 7))
#for i in range(len(N)):
#    pl.loglog(J, res[i], '.-', color=colors[i%len(colors)], 
#              markeredgecolor='k', label="N={0}".format(N[i]))
#pl.legend()
#pl.xlabel("J (number of terms in kernel)")
#pl.ylabel("time (seconds)")
#pl.savefig("plots/benchmarks_NJ.pdf")
