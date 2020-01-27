# -*- coding: utf-8 -*-

__all__ = ["RadiusImpact", "get_joint_radius_impact"]

import warnings

import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3.distributions import draw_values, generate_samples

from ..citations import add_citations_to_model
from ..utils import deprecated
from . import transforms as tr


class RadiusImpact(pm.Flat):
    """The Espinoza (2018) distribution over radius and impact parameter

    This is an implementation of `Espinoza (2018)
    <http://iopscience.iop.org/article/10.3847/2515-5172/aaef38/meta>`_
    The first axis of the shape of the parameter should be exactly 2. The
    radius ratio will be in the zeroth entry in the first dimension and
    the impact parameter will be in the first.

    Args:
        min_radius: The minimum allowed radius.
        max_radius: The maximum allowed radius.

    """

    __citations__ = ("espinoza18",)

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "'RadiusImpact' is deprecated. Use 'ImpactParameter' instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )

        add_citations_to_model(self.__citations__, kwargs.get("model", None))

        # Make sure that the shape is compatible
        shape = kwargs.get("shape", 2)
        try:
            if list(shape)[0] != 2:
                raise ValueError("the first dimension should be exactly 2")
        except TypeError:
            if shape != 2:
                raise ValueError("the first dimension should be exactly 2")

        self.min_radius = kwargs.pop("min_radius", 0)
        self.max_radius = kwargs.pop("max_radius", 1)
        transform = tr.radius_impact(self.min_radius, self.max_radius)
        kwargs["shape"] = shape
        kwargs["transform"] = transform

        super(RadiusImpact, self).__init__(*args, **kwargs)

        # Work out some reasonable starting values for the parameters
        default = np.zeros(shape)
        mn, mx = draw_values([self.min_radius - 0.0, self.max_radius - 0.0])
        default[0] = 0.5 * (mn + mx)
        default[1] = 0.5
        self._default = default

    def _random(self, pl, pu, size=None):
        r = np.moveaxis(
            np.random.uniform(0, 1, size=size), 0, -len(self.shape)
        )

        dr = pu - pl
        denom = 2 + pu + pl
        Ar = dr / denom

        r1 = r[0]
        r2 = r[1]
        m = r1 > Ar

        b = np.empty_like(r1)
        p = np.empty_like(r1)

        b[m] = (1 + pl) * (1 + (r1[m] - 1) / (1 - Ar))
        p[m] = pl + r2[m] * dr

        q1 = r1[~m] / Ar
        q2 = r2[~m]
        b[~m] = (1 + pl) + np.sqrt(q1) * q2 * dr
        p[~m] = pu - dr * np.sqrt(q1) * (1 - q2)

        pb = np.stack([p, b], axis=0)

        return np.moveaxis(pb, 0, -len(self.shape))

    def random(self, point=None, size=None):
        mn, mx = draw_values(
            [self.min_radius - 0.0, self.max_radius - 0.0], point=point
        )
        return generate_samples(
            self._random,
            mn,
            mx,
            dist_shape=self.shape,
            broadcast_shape=self.shape,
            size=size,
        )

    def logp(self, value):
        return tt.zeros_like(tt.as_tensor_variable(value))


@deprecated("ImpactParameter")
def get_joint_radius_impact(
    name="",
    N_planets=None,
    min_radius=0,
    max_radius=1,
    testval_r=None,
    testval_b=None,
    **kwargs,
):
    """Get the joint distribution over radius and impact parameter

    This uses the Espinoza (2018) parameterization of the distribution (see
    :class:`distributions.RadiusImpact` for more details).

    Args:
        name (Optional[str]): A prefix that is added to all distribution names
            used in this parameterization. For example, if ``name`` is
            ``param_``, vars will be added to the PyMC3 model with names
            ``param_rb`` (for the joint distribution), ``param_b``, and
            ``param_r``.
        N_planets (Optional[int]): The number of planets. If not provided, it
            will be inferred from the ``testval_*`` parameters or assumed to
            be 1.
        min_radius (Optional[float]): The minimum allowed radius.
        max_radius (Optional[float]): The maximum allowed radius.
        testval_r (Optional[float or array]): An initial guess for the radius
            parameter. This should be a ``float`` or an array with
            ``N_planets`` entries.
        testval_b (Optional[float or array]): An initial guess for the impact
            parameter. This should be a ``float`` or an array with
            ``N_planets`` entries.

    Returns:
        Two ``pymc3.Deterministic`` variables for the planet radius and impact
        parameter.

    """
    if N_planets is None:
        if testval_r is not None:
            N_planets = len(np.atleast_1d(testval_r))
        elif testval_b is not None:
            N_planets = len(np.atleast_1d(testval_b))
        else:
            N_planets = 1
    N_planets = int(N_planets)

    # Set up the testval for the rb parameter
    rb_test = np.zeros((2, N_planets))
    if testval_r is None:
        rb_test[0, :] = 0.5 * (min_radius + max_radius)
    else:
        rb_test[0, :] = testval_r
    if testval_b is None:
        rb_test[1, :] = 0.5
    else:
        rb_test[1, :] = testval_b

    # Construct the join distribution
    rb = RadiusImpact(
        name + "rb",
        min_radius=min_radius,
        max_radius=max_radius,
        shape=(2, N_planets),
        testval=rb_test,
        **kwargs,
    )

    # Extract the individual components
    r = pm.Deterministic(name + "r", rb[0])
    b = pm.Deterministic(name + "b", rb[1])

    return r, b
