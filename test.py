import numpy as np
import exoplanet as xo
import matplotlib.pyplot as pl
import utils
import itertools
import theano
from theano import tensor as tt

red = '#FE4365'
blue = '#00A9FF'
yellow = '#ECA25C'
green = '#3F9778'
darkblue = '#005D7F'

pl.rc('xtick', labelsize=20)
pl.rc('ytick', labelsize=20)
pl.rc('axes', labelsize=25)
pl.rc('axes', titlesize=30)
pl.rc('legend', fontsize=20)
pl.rc('lines', linewidth=4)

def fisher(t, tparams, gpparams, a=[1]):
    a = np.array(a)
    log_s, log_w0, log_q, diag = gpparams
    kernel = xo.gp.terms.SHOTerm(log_S0=log_s, log_w0=log_w0, log_Q=log_q)
    Q = a[:, None]*a[None, :]
    gp = xo.gp.GP(kernel=kernel, diag=diag*np.ones_like(t), x=t, J=2*len(a), Q=Q)
    dtrans = utils.transit(t, *tparams)
    x = tt.dmatrix()
    y = tt.dmatrix()
    f = theano.function([x, y], [x.T.dot(gp.apply_inverse_vector(y))])
    fish = np.zeros((4, 4))
    for (i, j) in itertools.product([0, 1, 2, 3], [0, 1, 2, 3]):
        x = np.array(dtrans[i+1])
        y = np.array(dtrans[j+1])
        x = np.tile(x, (np.shape(Q)[0], 1)).T.reshape(1, np.shape(Q)[0]*len(x)).T
        y = np.tile(y, (np.shape(Q)[0], 1)).T.reshape(1, np.shape(Q)[0]*len(y)).T
        fish[i, j] = f(x, y)[0][0, 0]
    return np.sqrt(np.diag(np.linalg.inv(fish)))

t = np.linspace(-8, 8, 1601)
tparams = [0.0, 0.1, 0.5, 0.05]  # t0, r, d, tin

alpha = -12 # total variance 
logr = np.linspace(-3, 3, 20)
logr = np.log(10 ** logr)
logsig = 0.5 * (alpha - np.log(1 + np.exp(2*logr)))
logs0 = logr + logsig
logq = np.log(1/np.sqrt(2))
diag = np.exp(2*logsig)
a = [1, 2]

w0T = 10
logw0 = np.log(w0T) - np.log(tparams[2])
logs0 = 2 * logs0  - logw0

oneband = [fisher(t, tparams, [ls + 2*np.log(np.mean(a)), logw0, logq, d]) for ls, d in zip(logs0, diag)]

oneband = np.array(oneband).T
x = np.log10(np.exp(logr))

pl.semilogy(x, oneband[0], 'r')
pl.semilogy(x, oneband[1], 'g')
pl.semilogy(x, oneband[2], 'b')
pl.savefig("test.pdf")
