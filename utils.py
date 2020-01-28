import numpy as np
import exoplanet as xo
import matplotlib.pyplot as pl
from jax import grad 

def transit(t, t0, r, d, tin):
    
    trans = np.zeros(len(t))
    dtdt0 = np.zeros(len(t))
    dtdr = np.zeros(len(t))
    dtdd = np.zeros(len(t))
    dtdtin = np.zeros(len(t))
    
    t1 = (t0-d/2-tin)
    t2 = (t0-d/2)
    t3 = (t0+d/2)
    t4 = (t0+d/2+tin)
    
    intransit = (t <= t3) & (t >= t2)
    ingress = (t < t2) & (t > t1)
    egress = (t < t4) & (t > t3)
    
    trans[intransit] = - (r**2)
    dtdt0[intransit] = 0
    dtdr[intransit] = -2*r
    dtdtin[intransit] = 0
    dtdd[intransit] = 0
    
    trans[ingress] = -(t[ingress] - (t0-d/2-tin)) * (r**2) / tin
    dtdt0[ingress] = (r**2) / tin
    dtdr[ingress] = -(t[ingress] - t1) * 2*r / tin
    dtdtin[ingress] = -(t0 - t[ingress] + d/2)*(r**2)/(tin**2)
    dtdd[ingress] = -(r**2)/(2*tin)
    
    trans[egress] = (1 - (r**2)) + (t[egress] - t3) * (r**2)/tin - 1
    dtdt0[egress] = -(r**2)/tin
    dtdr[egress] = -2*r + (t[egress] - t3) * 2 * r / tin
    dtdtin[egress] = -(t[egress] - t3) * (r**2) / (tin**2)
    dtdd[egress] = -(r**2)/(2*tin)
    
    return [trans, dtdt0, dtdr, dtdd, dtdtin]