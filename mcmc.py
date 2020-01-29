import pymc3 as pm
import theano.tensor as tt
from exoplanet.gp import terms, GP

with pm.Model() as model:
    logvar = pm.Normal("logvar", mu=2*np.log(sig), sd=prior_sig)

    # The parameters of the SHOTerm kernel
    logS0 = pm.Normal("logS0", mu=logs0_init, sd=5.0)
    logQ = pm.Normal("logQ", mu=logQ_init, sd=5.0)
    logw0 = pm.Normal("logw0", mu=logw0_init, sd=5.0)
    
    # The parameters for the transit mean function
    t0 = pm.Normal("t0", mu=t0_init, sd=5.0)
    BoundedNormal = pm.Bound(pm.Normal, lower=0.0, upper=1.0)
    r = BoundedNormal("r", mu=r_init, sd=1.0)
    d = pm.Normal("d", mu=d_init, sd=5.0)
    tin = pm.Normal("tin", mu=tin_init, sd=5.0)
            
    # Deterministics
    mean = pm.Deterministic("mean", utils.transit(t, t0, r, d, tin))

    # Set up the Gaussian Process model
    kernel = xo.gp.terms.RotationTerm(
        log_amp=logamp,
        period=period,
        log_Q0=logQ0,
        log_deltaQ=logdeltaQ,
        mix=mix
    )
    J = 4
            
    gp = xo.gp.GP(kernel, self.t, self.yerr**2 + tt.exp(logs2), J=J)

    # Compute the Gaussian Process likelihood and add it into the
    # the PyMC3 model as a "potential"
    pm.Potential("loglike", gp.log_likelihood([f - mean for f in flux]))

    # Compute the mean model prediction for plotting purposes
    #pm.Deterministic("mu", gp.predict())
    map_soln = xo.optimize(start=model.test_point, verbose=False)
