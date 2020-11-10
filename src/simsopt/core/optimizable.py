# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides classes and functions that are useful for
setting up optimizable objects and objective functions.
"""

import numpy as np
import types
import abc
import logging
import abc
from mpi4py import MPI

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

class Optimizable(metaclass=abc.ABCMeta):
    """
    Abstract base class provides useful features for optimizable functions.

    The class defines methods that are used by simsopt to know 
    degrees of freedoms (DOFs) associated with the optimizable
    function. All derived functions have to define get_dofs and
    set_dofs methods.
    """
    @abc.abstractmethod
    def get_dofs(self):
        """

        :return:
        """

    @abc.abstractmethod
    def set_dofs(self, x):
        """

        :param x:
        :return:
        """

    def index(self, dof_str):
        """
        Returns the index in the dof array whose name matches dof_str. 
        If not found, ValueError will be raised.
        """
        return self.dof_names.index(dof_str)

    def get_dof_by_name(self, dof_str):
        """
        Return a degree of freedom specified by its string name.
        """
        x = self.get_dofs()
        return x[self.index(dof_str)]

    def set_dof_by_name(self, dof_str, newval):
        """
        Set a degree of freedom specified by its string name.
        """
        x = self.get_dofs()
        x[self.index(dof_str)] = newval
        self.set_dofs(x)

    def is_dof_fixed(self, dof_str):
        """
        Identifies if the fixed attribute for a given DOF is set
        """
        return self.dof_fixed[self.index(dof_str)]
        
    def fix_dof(self, dof_str):
        """
        Set the fixed attribute for a given degree of freedom, specified by dof_str.
        """
        self.dof_fixed[self.index(dof_str)] = True

    def unfix_dof(self, dof_str):
        """
        Set the fixed attribute for a given degree of freedom, specified by dof_str.
        """
        self.dof_fixed[self.index(dof_str)] = False

    def fix_all_dofs(self):
        """
        Set the 'fixed' attribute for all degrees of freedom.
        """
        self.dof_fixed = np.full(len(self.get_dofs()), True)
        
    def unfix_all_dofs(self):
        """
        Set the 'fixed' attribute for all degrees of freedom.
        """
        self.dof_fixed = np.full(len(self.get_dofs()), False)
        
def function_from_user(target):
    """
    Given a user-supplied "target" to be optimized, extract the
    associated callable function.
    """
    if callable(target):
        return target
    elif hasattr(target, 'J') and callable(target.J):
        return target.J
    else:
        raise TypeError('Unable to find a callable function associated '
                        'with the user-supplied target ' + str(target))

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
    for method in ['index', 'get', 'set', 'get_fixed', 'set_fixed', 'all_fixed']:
        # See https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
        setattr(obj, method, types.MethodType(getattr(Optimizable, method), obj))

    return obj
