# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["StarryLightCurve"]

import numpy as np

import theano.tensor as tt

from .citations import add_citations_to_model
from .theano_ops.starry.get_cl import GetClOp
from .theano_ops.starry.limbdark import LimbDarkOp

get_cl = GetClOp()
limbdark = LimbDarkOp()


class StarryLightCurve(object):

    """A limb darkened light curve computed using starry

    Args:
        u (vector): A vector of limb darkening coefficients.
        r_star (Optional[scalar]): The stellar radius. If not given, this is
            assumed to be ``1`` so all coordinates should be given in stellar
            radii.

    """

    __citations__ = ("starry", )

    def __init__(self, u, r_star=1.0, model=None):
        add_citations_to_model(self.__citations__, model=model)

        self.r_star = tt.as_tensor_variable(r_star)
        self.u = tt.as_tensor_variable(u)
        u_ext = tt.concatenate([-1 + tt.zeros(1, dtype=self.u.dtype), self.u])
        self.c = get_cl(u_ext)
        self.c_norm = self.c / (np.pi * (self.c[0] + 2 * self.c[1] / 3))

    def get_light_curve(self, r, orbit, t, texp=None, oversample=7, order=2,
                        use_approx_in_transit=True, duration_factor=3):
        """Get the light curve for an orbit at a set of times

        Args:
            r (tensor): The radius of the transiting body in the same units as
                ``r_star``. This should have a shape that is consistent with
                the coordinates returned by ``orbit``. In general, this means
                that it should probably be a scalar or a vector with one entry
                for each body in ``orbit``.
            orbit: An object with a ``get_relative_position`` method that
                takes a tensor of times and returns a list of Cartesian
                coordinates of a set of bodies relative to the central source.
                This method should return three tensors (one for each
                coordinate dimension) and each tensor should have the shape
                ``append(t.shape, r.shape)`` or ``append(t.shape, oversample,
                r.shape)`` when ``texp`` is given. The first two coordinate
                dimensions are treated as being in the plane of the sky and the
                third coordinate is the line of sight with positive values
                pointing *away* from the observer. For an example, take a look
                at :class:`orbits.KeplerianOrbit`.
            t (tensor): The times where the light curve should be evaluated.
            texp (Optional[tensor]): The exposure time of each observation.
                This can be a scalar or a tensor with the same shape as ``t``.
                If ``texp`` is provided, ``t`` is assumed to indicate the
                timestamp at the *middle* of an exposure of length ``texp``.
            oversample (Optional[int]): The number of function evaluations to
                use when numerically integrating the exposure time.
            order (Optional[int]): The order of the numerical integration
                scheme. This must be one of the following: ``0`` for a
                centered Riemann sum (equivalent to the "resampling" procedure
                suggested by Kipping 2010), ``1`` for the trapezoid rule, or
                ``2`` for Simpson's rule.
            use_approx_in_transit (Optional[bool]): If ``True``, the model will
                only be evaluated for the data points expected to be in transit
                as computed using the ``approx_in_transit`` method on
                ``orbit``.

        """
        r = tt.as_tensor_variable(r)
        t = tt.as_tensor_variable(t)

        if use_approx_in_transit:
            transit_model = tt.zeros_like(t)
            inds = orbit.approx_in_transit(t, r=r, texp=texp,
                                           duration_factor=duration_factor)
            t = t[inds]

        if texp is None:
            tgrid = t
            rgrid = tt.shape_padleft(r, tgrid.ndim) \
                + tt.shape_padright(tt.zeros_like(tgrid), r.ndim)
        else:
            texp = tt.as_tensor_variable(texp)

            oversample = int(oversample)
            oversample += 1 - oversample % 2
            stencil = np.ones(oversample)

            # Construct the exposure time integration stencil
            if order == 0:
                dt = np.linspace(-0.5, 0.5, 2*oversample+1)[1:-1:2]
            elif order == 1:
                dt = np.linspace(-0.5, 0.5, oversample)
                stencil[1:-1] = 2
            elif order == 2:
                dt = np.linspace(-0.5, 0.5, oversample)
                stencil[1:-1:2] = 4
                stencil[2:-1:2] = 2
            else:
                raise ValueError("order must be <= 2")
            stencil /= np.sum(stencil)

            if texp.ndim == 0:
                dt = texp * dt
            else:
                dt = tt.shape_padright(texp) * dt
            tgrid = tt.shape_padright(t) + dt

            # Madness to get the shapes to work out...
            rgrid = tt.shape_padleft(r, tgrid.ndim + 1) \
                + tt.shape_padright(tt.zeros_like(tgrid), r.ndim)

        coords = orbit.get_relative_position(tgrid)
        b = tt.sqrt(coords[0]**2 + coords[1]**2)

        lc = self.compute_light_curve(
            b/self.r_star, rgrid/self.r_star, coords[2]/self.r_star)

        if texp is not None:
            stencil = tt.shape_padright(tt.shape_padleft(stencil, t.ndim),
                                        r.ndim)
            lc = tt.sum(stencil * lc, axis=t.ndim)
        lc = tt.sum(lc, axis=-1)

        if use_approx_in_transit:
            transit_model = tt.set_subtensor(transit_model[inds], lc)
            return transit_model
        else:
            return lc

    def compute_light_curve(self, b, r, los=None):
        """Compute the light curve for a set of impact parameters and radii

        .. note:: The stellar radius is *not* included in this method so the
            coordinates should be in units of the star's radius.

        Args:
            b (tensor): A tensor of impact parameter values.
            r (tensor): A tensor of radius ratios with the same shape as ``b``.
            los (Optional[tensor]): The coordinates of the body along the
                line-of-sight. If ``los < 0`` the body is between the observer
                and the source.

        """
        if los is None:
            los = -tt.ones_like(b)
        return limbdark(self.c_norm, b, r, los)