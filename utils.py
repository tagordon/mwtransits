import numpy as np
import exoplanet as xo
import matplotlib.pyplot as pl
import theano
from theano import tensor as tt

def transit(t, t0, r, d, tin):
    
    dtdt0 = np.zeros(len(t))
    dtdr = np.zeros(len(t))
    dtdd = np.zeros(len(t))
    dtdtin = np.zeros(len(t))
    
    t1 = (t0-d/2-tin)
    t2 = (t0-d/2)
    t3 = (t0+d/2)
    t4 = (t0+d/2+tin)
    
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

def theano_transit(t, t0, r, d, tin):
    
    t = tt.as_tensor_variable(t)
    trans = tt.zeros_like(t)
    
    t1 = (t0-d/2-tin)
    t2 = (t0-d/2)
    t3 = (t0+d/2)
    t4 = (t0+d/2+tin)
    
    #f1 = lambda t: [1, 0, 0, 0, 0]
    #f2 = lambda t: [1 - (t - (t0-d/2-tin)) * (r**2) / tin, 
    #                1 + (r**2)/tin, 
    #                1 - (t - t1) * 2*r / tin,
    #                1 - (t0 - t + d/2)*(r**2)/(tin**2),
    #                1- (r**2)/(2*tin)]
    #f3 = lambda t: [(r**2), 0, -2*r, 0, 0]
    #f4 = lambda t: [1 + (1 - (r**2)) + (t - t3) * (r**2)/tin - 1, 
    #                -(r**2)/tin, 
    #                -2*r + (t - t3) * 2 * r / tin,
    #                -(t - t3) * (r**2) / (tin**2),
    #                -(r**2)/(2*tin)]
    #f5 = lambda t: [1, 0, 0, 0, 0]
    
    f1 = lambda t: 1
    f2 = lambda t: 1 - (t - (t0-d/2-tin)) * (r**2) / tin
    f3 = lambda t: 1 - (r**2)
    f4 = lambda t: 1 + (1 - (r**2)) + (t - t3) * (r**2)/tin - 1
    f5 = lambda t: 1
    
    trans = tt.switch(tt.lt(t, t1), f1(t), 
                      tt.switch(tt.lt(t, t2), f2(t),
                                tt.switch(tt.lt(t, t3), f3(t), 
                                          tt.switch(tt.lt(t, t4), f4(t), 
                                                    f5(t))
                                         )
                               )
                     )
    return trans    