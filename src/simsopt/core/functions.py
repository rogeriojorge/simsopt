# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a few minimal optimizable objects, each
representing a function. These functions are mostly used for testing.
"""

import numpy as np
import logging
import deprecated
from mpi4py import MPI
from typing import Union
from .optimizable import Optimizable, DOFs, DOFsDataFrame
from .util import Real, RealArray

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)


# Alternative Implementation of Identity
class Identity(Optimizable):
    def __init__(self,
                 x: Real = 0.0,
                 dof_name: str = None,
                 dof_fixed: bool = False):
        super().__init__([x],
                         [dof_name] if dof_name is not None else None,
                         [dof_fixed])

    def f(self):
        return self.x

    def dJ(self, x: RealArray = None):
        if x is not None:
            if isinstance(x, Real):
                self.x = [x]
            else:
                self.x = x
        return np.array([1.0])


class Adder(Optimizable):
    """This class defines a minimal object that can be optimized. It has
    n degrees of freedom, and has a function that just returns the sum
    of these dofs. This class is used for testing.
    """
    def __init__(self, n=3, x0=None, dof_names=None):
        self.n = n
        x = x0 if x0 is not None else np.zeros(n)
        super().__init__(x, names=dof_names)

    #def __call__(self, x: RealArray = None):
    #    if x is not None:
    #        self.x = x
    #    if self.new_x:
    #        self._val = np.sum(self._dofs.full_x)
    #        self.new_x = False
    #    return self._val
    def f(self):
        return np.sum(self._dofs.full_x)

    def dJ(self):
        return np.ones(self.n)
        
    @property
    def df(self):
        """
        Same as the function dJ(), but a property instead of a function.
        """
        return self.dJ()


class Rosenbrock(Optimizable):
    """
    This class defines a minimal object that can be optimized.
    """
    def __init__(self, b=100.0, x=0.0, y=0.0):
        self._sqrtb = np.sqrt(b)
        super().__init__([x, y], names=['x', 'y'])

    @property
    def term1(self):
        """
        Returns the first of the two quantities that is squared and summed.
        """
        #return self._x - 1
        return self._dofs.full_x[0] - 1

    @property
    def term2(self):
        """
        Returns the second of the two quantities that is squared and summed.
        """
        x = self._dofs.full_x[0]
        y = self._dofs.full_x[1]
        return (x * x - y) / self._sqrtb

    @property
    def dterm1(self):
        """
        Returns the gradient of term1
        """
        return np.array([1.0, 0.0])

    @property
    def dterm2(self):
        """
        Returns the gradient of term2
        """
        return np.array([2 * self._dofs.full_x[0], -1.0]) / self._sqrtb
    
    def f(self):
        """
        Returns the total function, squaring and summing the two terms.
        """
        t1 = self.term1
        t2 = self.term2
        return t1 * t1 + t2 * t2

    def terms(self):
        """
        Returns term1 and term2 together as a 2-element numpy vector.
        """
        return np.array([self.term1(), self.term2()])

    def dterms(self):
        """
        Returns the 2x2 Jacobian for term1 and term2.
        """
        return np.array([[1.0, 0.0],
                         [2 * self._x / self._sqrtb, -1.0 / self._sqrtb]])


class TestObject1(Optimizable):
    """
    This is an optimizable object used for testing. It depends on two
    sub-objects, both of type Adder.
    """
    def __init__(self, val):
        self.val = val
        self.dof_names = ['val']
        self.dof_fixed = np.array([False])
        self.adder1 = Adder(3)
        self.adder2 = Adder(2)
        self.depends_on = ['adder1', 'adder2']

    def set_dofs(self, x):
        self.val = x[0]

    def get_dofs(self):
        return np.array([self.val])
    
    def J(self):
        return (self.val + 2 * self.adder1.J()) / (10.0 + self.adder2.J())

    def dJ(self):
        v = self.val
        a1 = self.adder1.J()
        a2 = self.adder2.J()
        # J = (v + 2 * a1) / (10 + a2)
        return np.concatenate((np.array([1.0 / (10.0 + a2)]),
                               np.full(self.adder1.n, 2.0 / (10.0 + a2)),
                               np.full(self.adder2.n, -(v + 2 * a1) / ((10.0 + a2) ** 2))))
    @property
    def f(self):
        """
        Same as J() but a property instead of a function.
        """
        return self.J()

    @property
    def df(self):
        """
        Same as dJ() but a property instead of a function.
        """
        return self.dJ()

    
class TestObject2(Optimizable):
    """
    This is an optimizable object used for testing. It depends on two
    sub-objects, both of type Adder.
    """
    def __init__(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
        self.dof_names = ['val1', 'val2']
        self.dof_fixed = np.array([False, False])
        self.t = TestObject1(0.0)
        self.adder = Adder(2)
        self.depends_on = ['t', 'adder']

    def set_dofs(self, x):
        self.val1 = x[0]
        self.val2 = x[1]

    def get_dofs(self):
        return np.array([self.val1, self.val2])
    
    def J(self):
        v1 = self.val1
        v2 = self.val2
        t = self.t.J()
        a = self.adder.J()
        return v1 + a * np.cos(v2 + t)

    def dJ(self):
        v1 = self.val1
        v2 = self.val2
        a = self.adder.J()
        t = self.t.J()
        cosat = np.cos(v2 + t)
        sinat = np.sin(v2 + t)
        # Order of terms in the gradient: v1, v2, t, a
        return np.concatenate((np.array([1.0, -a * sinat]),
                               -a * sinat * self.t.dJ(),
                               cosat * self.adder.dJ()))
    @property
    def f(self):
        """
        Same as J() but a property instead of a function.
        """
        return self.J()

    @property
    def df(self):
        """
        Same as dJ() but a property instead of a function.
        """
        return self.dJ()

    
class Affine(Optimizable):
    """
    This class represents a random affine (i.e. linear plus constant)
    transformation from R^n to R^m.
    """
    def __init__(self, nparams, nvals):
        """
        nparams = number of independent variables.
        nvals = number of dependent variables.
        """
        self.nparams = nparams
        self.nvals = nvals
        self.A = (np.random.rand(nvals, nparams) - 0.5) * 4
        self.B = (np.random.rand(nvals) - 0.5) * 4
        super().__init__(np.zeros(nparams))

    def f(self):
        return np.matmul(self.A, self.full_x) + self.B

    def dJ(self):
        return self.A
    
