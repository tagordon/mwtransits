import pymc3 as pm
import theano.tensor as tt
from exoplanet.gp import terms, GP
import numpy as np
import utils
import exoplanet as xo
import matplotlib.pyplot as pl

# make corresponding two-band and one-band datasets 
def make_data(t, log_S0, log_w0, log_Q, logsig, 
              t0=5.0, r=0.1, d=1.0, tin=0.1, a=2.0):
    
    transit = utils.theano_transit(t, 
                                   t0, 
                                   r, 
                                   d, 
                                   tin)
    kernel = xo.gp.terms.SHOTerm(
        log_S0 = log_S0,
        log_w0 = log_w0,
        log_Q=log_Q
    )
    J = 4
    
    q = np.array([1, a])
    Q = q[:, None]*q[None, :]
    diag = np.exp(2*logsig)*np.ones((2, len(t)))
    gp = GP(kernel, t, diag, J=J, Q=Q)
    n = gp.dot_l(np.random.randn(2*len(t), 1)).eval()
    transit = tt.reshape(tt.tile(transit, (2, 1)).T, 
                         (1, 2*transit.shape[0])).T
    n += transit.eval()
    return n, np.sum([n[i::2].T[0] for i in range(len(q))], axis=0)/2

def run_mcmc_2d(t, data, logS0_init, 
                logw0_init, logQ_init, 
                logsig_init, t0_init, 
                r_init, d_init, tin_init, 
                a_init):
    
    with pm.Model() as model:
        #logsig = pm.Uniform("logsig", lower=-20.0, upper=0.0, testval=logsig_init)

        # The parameters of the SHOTerm kernel
        #logS0 = pm.Uniform("logS0", lower=-50.0, upper=0.0, testval=logS0_init)
        #logQ = pm.Uniform("logQ", lower=-50.0, upper=20.0, testval=logQ_init)
        #logw0 = pm.Uniform("logw0", lower=-50.0, upper=20.0, testval=logw0_init)
    
        a = pm.Uniform("a", lower=1.0, upper=10.0, testval=2.0)
    
        # The parameters for the transit mean function
        t0 = pm.Uniform("t0", lower=t[0], upper=t[-1], testval=t0_init)
        r = pm.Uniform("r", lower=0.0, upper=1.0, testval=r_init)
        d = pm.Uniform("d", lower=0.0, upper=10.0, testval=d_init)
        tin = pm.Uniform("tin", lower=0.0, upper=10.0, testval=tin_init)
            
        # Deterministics
        # mean = pm.Deterministic("mean", utils.transit(t, t0, r, d, tin))
        transit = utils.theano_transit(t, t0, r, d, tin)
        transit = tt.reshape(tt.tile(transit, (2, 1)).T, (1, 2*transit.shape[0])).T

        # Set up the Gaussian Process model
        kernel = xo.gp.terms.SHOTerm(
            log_S0 = logS0_init,
            log_w0 = logw0_init,
            log_Q=logQ_init
        )
    
        q = tt.stack(1, a)
        Q = q[:, None]*q[None, :]
        diag = np.exp(2*logsig_init)*tt.ones((2, len(t)))
        gp = GP(kernel, t, diag, J=4, Q=Q)

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        pm.Potential("loglike", gp.log_likelihood((data - transit).T))

        # Compute the mean model prediction for plotting purposes
        #pm.Deterministic("mu", gp.predict())
        map_soln = xo.optimize(start=model.test_point, verbose=False)
        
    with model:
        map_soln = xo.optimize(start=model.test_point)
        
    with model:
        trace = pm.sample(
            tune=500,
            draws=500,
            start=map_soln,
            cores=2,
            chains=2,
            step=xo.get_dense_nuts_step(target_accept=0.9),
        )
    return trace

def run_mcmc_1d(t, data, logS0_init, 
                logw0_init, logQ_init, 
                logsig_init, t0_init, 
                r_init, d_init, tin_init):
    
    with pm.Model() as model:
        #logsig = pm.Uniform("logsig", lower=-20.0, upper=0.0, testval=logsig_init)

        # The parameters of the SHOTerm kernel
        #logS0 = pm.Uniform("logS0", lower=-50.0, upper=0.0, testval=logS0_init)
        #logQ = pm.Uniform("logQ", lower=-50.0, upper=20.0, testval=logQ_init)
        #logw0 = pm.Uniform("logw0", lower=-50.0, upper=20.0, testval=logw0_init)
        
        # The parameters for the transit mean function
        t0 = pm.Uniform("t0", lower=t[0], upper=t[-1], testval=t0_init)
        r = pm.Uniform("r", lower=0.0, upper=1.0, testval=r_init)
        d = pm.Uniform("d", lower=0.0, upper=10.0, testval=d_init)
        tin = pm.Uniform("tin", lower=0.0, upper=10.0, testval=tin_init)
            
        # Deterministics
        # mean = pm.Deterministic("mean", utils.transit(t, t0, r, d, tin))
        transit = utils.theano_transit(t, t0, r, d, tin)

        # Set up the Gaussian Process model
        kernel = xo.gp.terms.SHOTerm(
            log_S0 = logS0_init,
            log_w0 = logw0_init,
            log_Q=logQ_init
        )
    
        diag = np.exp(2*logsig_init)*tt.ones((1, len(t)))
        gp = GP(kernel, t, diag, J=2)

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        pm.Potential("loglike", gp.log_likelihood(data - transit))

        # Compute the mean model prediction for plotting purposes
        #pm.Deterministic("mu", gp.predict())
        map_soln = xo.optimize(start=model.test_point, verbose=False)
        
    with model:
        map_soln = xo.optimize(start=model.test_point)
        
    with model:
        trace = pm.sample(
            tune=500,
            draws=500,
            start=map_soln,
            cores=2,
            chains=2,
            step=xo.get_dense_nuts_step(target_accept=0.9),
        )
    return trace
    
def load_models(t, data1, data2, logS0_init, 
                logw0_init, logQ_init, 
                logsig_init, t0_init, 
                r_init, d_init, tin_init, a):
    
    with pm.Model() as model1d:
        logsig = pm.Uniform("logsig", lower=-20.0, upper=0.0, testval=logsig_init)

        # The parameters of the SHOTerm kernel
        #logS0 = pm.Uniform("logS0", lower=-50.0, upper=0.0, testval=logS0_init)
        #logQ = pm.Uniform("logQ", lower=-50.0, upper=20.0, testval=logQ_init)
        #logw0 = pm.Uniform("logw0", lower=-50.0, upper=20.0, testval=logw0_init)
        
        # The parameters for the transit mean function
        t0 = pm.Uniform("t0", lower=t[0], upper=t[-1], testval=t0_init)
        r = pm.Uniform("r", lower=0.0, upper=1.0, testval=r_init)
        d = pm.Uniform("d", lower=0.0, upper=10.0, testval=d_init)
        tin = pm.Uniform("tin", lower=0.0, upper=10.0, testval=tin_init)
            
        # Deterministics
        # mean = pm.Deterministic("mean", utils.transit(t, t0, r, d, tin))
        transit = utils.theano_transit(t, t0, r, d, tin)

        # Set up the Gaussian Process model
        kernel = xo.gp.terms.SHOTerm(
            log_S0 = logS0_init,
            log_w0 = logw0_init,
            log_Q=logQ_init
        )
    
        diag = np.exp(2*logsig_init)*tt.ones((1, len(t)))
        gp = GP(kernel, t, diag, J=2)

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        pm.Potential("loglike", gp.log_likelihood(data1 - transit))

        # Compute the mean model prediction for plotting purposes
        #pm.Deterministic("mu", gp.predict())
        map_soln = xo.optimize(start=model1d.test_point, verbose=False)
        
    with pm.Model() as model2d:
        #logsig = pm.Uniform("logsig", lower=-20.0, upper=0.0, testval=logsig_init)

        # The parameters of the SHOTerm kernel
        #logS0 = pm.Uniform("logS0", lower=-50.0, upper=0.0, testval=logS0_init)
        #logQ = pm.Uniform("logQ", lower=-50.0, upper=20.0, testval=logQ_init)
        #logw0 = pm.Uniform("logw0", lower=-50.0, upper=20.0, testval=logw0_init)
    
        a = pm.Uniform("a", lower=1.0, upper=10.0, testval=2.0)
    
        # The parameters for the transit mean function
        t0 = pm.Uniform("t0", lower=t[0], upper=t[-1], testval=t0_init)
        r = pm.Uniform("r", lower=0.0, upper=1.0, testval=r_init)
        d = pm.Uniform("d", lower=0.0, upper=10.0, testval=d_init)
        tin = pm.Uniform("tin", lower=0.0, upper=10.0, testval=tin_init)
            
        # Deterministics
        # mean = pm.Deterministic("mean", utils.transit(t, t0, r, d, tin))
        transit = utils.theano_transit(t, t0, r, d, tin)
        transit = tt.reshape(tt.tile(transit, (2, 1)).T, (1, 2*transit.shape[0])).T

        # Set up the Gaussian Process model
        kernel = xo.gp.terms.SHOTerm(
            log_S0 = logS0_init,
            log_w0 = logw0_init,
            log_Q=logQ_init
        )
    
        q = tt.stack(1, a)
        Q = q[:, None]*q[None, :]
        diag = np.exp(2*logsig_init)*tt.ones((2, len(t)))
        gp = GP(kernel, t, diag, J=4, Q=Q)

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        pm.Potential("loglike", gp.log_likelihood((data2 - transit).T))

        # Compute the mean model prediction for plotting purposes
        #pm.Deterministic("mu", gp.predict())
        map_soln = xo.optimize(start=model2d.test_point, verbose=False)
    return model1d, model2d