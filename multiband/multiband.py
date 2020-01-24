import numpy as np
from scipy import linalg
import exoplanet as xo

class multigp_sho:
    
    def __init__(self, log_S0, log_w0, log_Q, sig, a):
        self.log_S0=log_S0
        self.log_w0=log_w0
        self.log_Q=log_Q
        self.sig=sig
        self.a=a
        self.term = xo.gp.terms.SHOTerm(log_S0=log_S0, 
                                     log_w0=log_w0,
                                     log_Q=log_Q)

    def get_coefficients(self):
        log_eta = 0.5*np.log(4*np.exp(2*self.log_Q) - 1)
        log_a = self.log_S0 + self.log_w0 + self.log_Q
        log_b = log_a - log_eta
        log_c = self.log_w0 - self.log_Q - np.log(2)
        log_d = log_c + log_eta
        return np.exp([log_a, log_b, log_c, log_d])

    def evaluate_kernel(self, tau):
        a, b, c, d = self.get_coefficients()
        return a*np.exp(-c*tau)*np.cos(d*tau) + b*np.exp(-c*tau)*np.sin(d*tau)

    def get_1d_matrix(self, t, sig):
        m = np.vectorize(self.evaluate_kernel)(np.abs(t[None, :] - t[:, None]))
        return np.eye(len(t))*sig + m

    def get_2d_matrix(self, t):
        Q = self.a[None, :]*self.a[:, None]
        m = np.kron(Q, self.get_1d_matrix(t, 0.0))
        return np.diag(np.kron(self.sig*self.sig, np.ones(len(t)))) + m

    def apply_inverse_direct(self, t, r):
        return linalg.inv(self.get_2d_matrix(t)).dot(r)
    
    def multiply_direct(self, t, r):
        return self.get_2d_matrix(t).dot(r)
    
    def multiply_celerite(self, t, r):
        a, U, V, P = self.term.get_celerite_matrices(t, 0.0)
        a, U, V, P = a.eval(), U.eval(), V.eval(), P.eval()
        fp = np.zeros(len(t), 2)
        for i in range(len(t), 0, -1):
            fp[i] = P[i+1]*(fp[i+1]+U[i]*r[i+1])
        fm = np.zeros(len(t), 2)
        for i in range(len(t)-1):
            fm = P[i]*(f[i-1]+V[i-1]*z[i-1])
        y = a*z + np.sum(V*fp + U*fm)
    
    #def apply_inverse_eric(t, r):
    #    alpha = self.a/(self.sig*self.sig)
    #    rs = np.sum(alpha*r)
    #    alpha = np.sum(alpha*self.a)
    #    shoterm1 = xo.gp.terms.SHOTerm(log_S0=self.log_S0, 
    #                                 log_w0=self.log_w0,
    #                                 log_Q=self.log_Q)
    #    shoterm2 = xo.gp.terms.SHOTerm(log_S0=alpha*self.log_S0, 
    #                                 log_w0=self.log_w0,
    #                                 log_Q=self.log_Q)
    #    gp1 = xo.gp.GP(shoterm1, t, 0.0)
    #    gp2 = xo.gp.GP(shoterm2, t, 1.0)
    #    zs = 
        
    
    
    