# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides classes and functions that are useful for
setting up optimizable objects and objective functions.
"""

import numpy as np
import types
import logging
import abc
from mpi4py import MPI
from deprecated import deprecated
from collections.abc import Callable

from .dofs import DOF, DOFs


logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

class Optimizable(Callable):
    """
    Callable ABC that provides useful features for optimizable objects.

    The class defines methods that are used by simsopt to know 
    degrees of freedoms (DOFs) associated with the optimizable
    objects. All derived objects have to implement __call__ and contains
    _dofs member which is an instance of DOFs class.
    """
    @deprecated(version='0.0.2', reason="You should use dofs property")
    def get_dofs(self):
        """

        :return:
        """
        return self.dofs

    @deprecated(version='0.0.2', reason="You should use dofs property")
    def set_dofs(self, x):
        """

        Args:
            x: State vector as a Sequence
        """
        self.dofs = x

    @property
    def dofs(self):
        return [dof.x for dof in self._dofs if dof.is_free()]

    @dofs.setter
    def dofs(self, x):
        i = 0
        for dof in self._dofs:
            if dof.is_free():
                dof.x = x[i]
                i += 1
        else:
            if i < len(x):
                raise IndexError(
                    "Size of state vector mismatches with free DOFs")

    @property
    def state(self):
        return self.dofs

    def dof_index(self, key):
        """
        Returns the index in the dof array whose name matches dof_str. 
        If not found, ValueError will be raised.
        """
        if isinstance(key, str):
            dof = self._dofs[key]
        elif isinstance(key, DOFs):
            dof = key
        return self._dofs.index(dof)

    def get_dof(self, key):
        """
        Return a degree of freedom specified by its name or by index.
        """
        return self._dofs[key].x

    def set_dof(self, key, new_val):
        """
        Set a degree of freedom specified by its name or by index.
        """
        self._dofs[key] = new_val

    def is_dof_fixed(self, key):
        """
        Tells if the dof specified with its name or by index is fixed
        """
        return self._dofs[key].is_fixed()

    def is_dof_free(self, key):
        """
        Tells if the dof specified with its name or by index is fixed
        """
        return self._dofs[key].is_free()

    def fix_dof(self, key):
        """
        Set the fixed attribute for a given degree of freedom, specified by dof_str.
        """
        self._dofs[key].fix()

    def unfix_dof(self, key):
        """
        Set the fixed attribute for a given degree of freedom, specified by dof_str.
        """
        self._dofs[key].unfix()

    def fix_all_dofs(self):
        """
        Set the 'fixed' attribute for all degrees of freedom.
        """
        #self.dof_fixed = np.full(len(self.get_dofs()), True)
        self._dofs.fix_all()

    def unfix_all_dofs(self):
        """
        Set the 'fixed' attribute for all degrees of freedom.
        """
        #self.dof_fixed = np.full(len(self.get_dofs()), False)
        self._dofs.unfix_all()


class Target(Optimizable):
    """
    Given an attribute of an object, which typically would be a
    @property, form a callable function that can be used as a target
    for optimization.
    """
    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr
        self.depends_on = ["obj"]
        
        # Attach a dJ function only if obj has one
        def dJ(self0):
            return getattr(self0.obj, 'd' + self0.attr)
        if hasattr(obj, 'd' + attr):
            self.dJ = types.MethodType(dJ, self)

        self.__call__ = lambda x: obj.attr
        
    def J(self):
        return getattr(self.obj, self.attr)

    #def dJ(self):
    #    return getattr(self.obj, 'd' + self.attr)

    def get_dofs(self):
        return np.array([])

    def set_dofs(self, v):
        pass


def optimizable(obj):
    """
    Given any object, add attributes like fixed, mins, and maxs. fixed
    = False by default. Also, add the other methods of Optimizable to
    the object.
    """

    # If the object does not have a get_dofs() method, attach one,
    # assuming the object does not directly own any dofs.
    def get_dofs(self):
        return np.array([])
    def set_dofs(self, x):
        pass
    if not hasattr(obj, 'get_dofs'):
        obj.get_dofs = types.MethodType(get_dofs, obj)
    if not hasattr(obj, 'set_dofs'):
        obj.set_dofs = types.MethodType(set_dofs, obj)
            
    n = len(obj.get_dofs())
    if not hasattr(obj, 'dof_fixed'):
        obj.dof_fixed = np.full(n, False)
    if not hasattr(obj, 'mins'):
        obj.mins = np.full(n, np.NINF)
    if not hasattr(obj, 'maxs'):
        obj.maxs = np.full(n, np.Inf)

    # Add the following methods from the Optimizable class:
    #for method in ['index', 'get', 'set', 'get_fixed', 'set_fixed', 'all_fixed']:
        # See https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
        #setattr(obj, method, types.MethodType(getattr(Optimizable, method), obj))

    # New compact implementation
    method_list = [f for f in dir(Optimizable) if \
            callable(getattr(Optimizable, f)) and not f.startswith("__")]
    for f in method_list:
        if not hasattr(obj, f) and f not in ('get_dofs', 'set_dofs'):
            setattr(obj, f, types.MethodType(getattr(Optimizable, f), obj))

    return obj
