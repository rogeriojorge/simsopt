# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides classes and functions that are useful for
setting up optimizable objects and objective functions.
"""

from __future__ import annotations

import abc
import copy
import logging
import types
import hashlib
from collections.abc import Callable, Hashable, Sequence, MutableSequence
from numbers import Real, Integral
from typing import Union, Any, Tuple

import numpy as np
import pandas as pd
from mpi4py import MPI
from deprecated import deprecated
#from monty.json import MSONable

from .util import unique, Array, RealArray, StrArray, BoolArray, Key, IntArray
from .util import ImmutableId, InstanceCounterABCMeta

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

# Types
  # To denote arguments for accessing individual dof

def get_owners(obj, owners_so_far=[]):
    """
    Given an object, return a list of objects that own any
    degrees of freedom, including both the input object and any of its
    dependendents, if there are any.
    """
    owners = [obj]
    # If the 'depends_on' attribute does not exist, assume obj does
    # not depend on the dofs of any other objects.
    if hasattr(obj, 'depends_on'):
        for j in obj.depends_on:
            subobj = getattr(obj, j)
            if subobj in owners_so_far:
                raise RuntimeError('Circular dependency detected among the objects')
            owners += get_owners(subobj, owners_so_far=owners)
    return owners


def function_from_user(target):
    """
    Given a user-supplied "target" to be optimized, extract the
    associated callable function.
    """
    if callable(target):
        return target
    # The method J is to conform with sismgeo
    elif hasattr(target, 'J') and callable(target.J):
        return target.J
    else:
        raise TypeError('Unable to find a callable function associated '
                        'with the user-supplied target ' + str(target))


class DOF:
    """
    A generalized class to represent an individual degrees of freedom
    associated with optimizable functions.
    """

    def __init__(self,
                 x: Real,
                 name: str,
                 free: bool = True,
                 lower_bound: Real = np.NINF,
                 upper_bound: Real = np.Inf):
        """

        :param name: Name of DOF for easy reference
        :param fixed: Flag to denote if the ODF is constrained?
        :param lower_bound: Minimum allowed value of DOF
        :param upper_bound: Maximum allowed value of DOF
        """
        self._x = x
        self.name = name
        self._free = free
        self._lb = lower_bound
        self._ub = upper_bound

    #def __hash__(self):
    #    return hash(":".join(map(str, [self.owner, self.name])))

    #@property
    #def extended_name(self) -> str:
    #    return ":".join(map(str, [self.owner, self.name]))

    def __repr__(self) -> str:
        return "DOF: {}, value = {}, fixed = {}, bounds = ({}, {})".format(
            self.name, self._x, not self._free, self._lb, self._ub)

    def __eq__(self, other):
        return all([self.name == other.name,
                   np.isclose(self._x, other._x),
                   self._free == other._free,
                   np.isclose(self._lb, other._lb),
                   np.isclose(self._ub, other._ub)])

    def is_fixed(self) -> bool:
        """
        Checks ifs the DOF is fixed

        Returns:
            True if DOF is fixed else False
        """
        return not self._free

    def is_free(self) -> bool:
        """
        Checks ifs the DOF is fixed

        Returns:
            True if DOF is fixed else False
        """
        return self._free

    def fix(self) -> None:
        """
        Denotes that the DOF needs to be fixed during optimization
        """
        self._free = False

    def unfix(self):
        """
        Denotes that the DOF can be varied during optimization
        """
        self._free = True

    @property
    def min(self) -> Real:
        """
        Minimum value of DOF allowed

        :return: Lower bound of DOF if not fixed else None
        """
        return self._lb if self._free else None

    @min.setter
    def min(self, lower_bound):
        self._lb = lower_bound

    @property
    def max(self) -> Real:
        """
        Maximum value of DOF allowed

        :return: Upper bound of DOF if not fixed else None
        """
        return self._ub if self._free else None

    @max.setter
    def max(self, upper_bound: Real) -> None:
        self._ub = upper_bound

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if self.is_fixed():
            raise TypeError("Updating state is forbidded for fixed DOF")
        if x > self.max or x < self.min:
            raise ValueError(
                "Input state is out of bounds for the DOF {}".format(self.name))
        self._x = x


class DOFsDataFrame(pd.DataFrame):
    """
    Defines the (D)egrees (O)f (F)reedom(s) for optimization

    This class holds data related to the vector of degrees of freedom
    that have been combined from multiple optimizable objects.
    """
    def __init__(self,
                 x: RealArray = None, # To enable empty DOFs object
                 names: StrArray = None,
                 free: BoolArray = None,
                 lower_bounds: RealArray = None,
                 upper_bounds: RealArray = None):
        """
        Args:
            owners: Objects owning the dofs
            names: Names of the dofs
            x: Values of the dofs
            fixed: Array of boolean values denoting if the DOFs is fixed
            lower_bounds: Lower bounds for the DOFs. Meaningful only if
                DOF is not fixed. Default is np.NINF
            upper_bounds: Upper bounds for the DOFs. Meaningful only if
                DOF is not fixed. Default is np.inf
        """
        if x is None:
            x = np.array([])
        else:
            x = np.array(x, dtype=np.float)
        if names is None:
            names = ["x{}".format(i) for i in range(len(x))]

        if free is not None:
            free = np.array(free, dtype=np.bool_)
        else:
            free = np.full(len(x), True)

        if lower_bounds is not None:
            lb = np.array(lower_bounds, np.float)
        else:
            lb = np.full(len(x), np.NINF)

        if upper_bounds is not None:
            ub = np.array(upper_bounds, np.float)
        else:
            ub = np.full(len(x), np.inf)
        super().__init__(data={"_x": x, "free": free, "_lb": lb, "_ub": ub},
                         index=names)

    def fix(self, key: Key) -> None:
        if isinstance(key, str):
            self.loc[key, 'free'] = False
        else:
            self.free.iloc[key] = False

    def unfix(self, key: Key) -> None:
        if isinstance(key, str):
            self.loc[key, 'free'] = True
        else:
            self.free.iloc[key] = True

    def fix_all(self) -> None:
        self.free.apply(lambda x: False)

    def unfix_all(self):
        """
        Make vall DOFs variable
        Caution: Make sure the bounds are well defined
        Returns:

        """
        self.free.apply(lambda x: True)

    def any_free(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if any free DOF is found
        """
        return self.free.any()

    def any_fixed(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if any fixed DOF is found
        """
        return not self.free.all()

    def all_free(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if all DOFs are free to vary
        """
        return self.free.all()

    def all_fixed(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if all DOFs are fixed
        """
        return not self.free.any()

    @property
    def x(self) -> RealArray:
        return self._x[self.free].to_numpy()

    @x.setter
    def x(self, x: RealArray) -> None:
        """

        Args:
            x: Array of values to set x
            Note: This method blindly broadcasts a single value.
            So don't supply a single value unless you really desire
        """
        self.loc[self.free, "_x"] = x

    @property
    def full_x(self) -> RealArray:
        """
        Return all x even the fixed ones

        Returns:
            Pruned DOFs object without any fixed DOFs
        """
        return self._x.to_numpy()

    @property
    def reduced_len(self) -> Integral:
        """
        The standard len function returns the full length of DOFs.
        To get the number of free DOFs use DOFs._reduced_len method
        Returns:

        """
        return len(self.free[self.free])

    @property
    def lower_bounds(self) -> RealArray:
        """

        Returns:

        """
        return self._lb[self.free].to_numpy()

    @lower_bounds.setter
    def lower_bounds(self, lower_bounds: RealArray):
        self.loc[self.free, "_lb"] = lower_bounds

    @property
    def upper_bounds(self) -> RealArray:
        return self._ub[self.free].to_numpy()

    @upper_bounds.setter
    def upper_bounds(self, upper_bounds: RealArray):
        self.loc[self.free, "_ub"] = upper_bounds

    @property
    def bounds(self) -> Tuple[RealArray, RealArray]:
        return (self.lower_bounds, self.upper_bounds)

    def update_upper_bound(self, key: Key, val: Real):
        if isinstance(key, str):
            self._ub[key] = val
        else:
            self._ub.iloc[key] = val

    def update_lower_bound(self, key: Key, val: Real):
        if isinstance(key, str):
            self._lb[key] = val
        else:
            self._lb.iloc[key] = val

    def update_bounds(self, key: Key, val: Tuple[Real, Real]):
        if isinstance(key, str):
            self._lb[key] = val[0]
            self._ub[key] = val[1]
        else:
            self._lb.iloc[key] = val[0]
            self._ub.iloc[key] = val[1]


class DOFs(MutableSequence):
    """
    Defines the (D)egrees (O)f (F)reedom(s) for optimization

    This class holds data related to the vector of degrees of freedom
    that have been combined from multiple optimizable objects.
    """
    def __init__(self,
                 x: RealArray,
                 names: StrArray = None,
                 free: BoolArray = None,
                 lower_bounds: RealArray = None,
                 upper_bounds: RealArray = None):
        """
        Args:
            names: Names of the dofs
            x: Values of the dofs
            fixed: Array of boolean values denoting if the DOFs is fixed
            lower_bounds: Lower bounds for the DOFs. Meaningful only if
                DOF is not fixed. Default is np.NINF
            upper_bounds: Upper bounds for the DOFs. Meaningful only if
                DOF is not fixed. Default is np.inf
        """
        self._x = np.array(x)
        if names is not None:
            self.names = np.array(names)
        else:
            self.names = np.array(map(lambda i: "x[{}]".format(i),
                                      range(len(self._x))))

        if free is not None:
            self.free = np.array(free)
        else:
            self.free = np.full(len(self._x), True)

        if lower_bounds is not None:
            self._lb = np.array(lower_bounds)
        else:
            self._lb = np.full(len(self._x), np.NINF)

        if upper_bounds is not None:
            self._ub = np.array(upper_bounds)
        else:
            self._ub = np.full(len(self._x), np.inf)

    # Magic method associated with MutableSequence
    def __getitem__(self, key: Key) -> DOF:
        if isinstance(key, Integral):
            i = key
        elif isinstance(key, str): # Implement string indexing
            i, = np.where(self.names == key)
            if not i.size:
                raise IndexError("Element not found.")
        else:
            raise TypeError("Wrong type as index. Supply either an integer"
                            " or string for index.")

        return DOF(self.names[i], self._x[i], self.free[i],
                   self._lb[i], self._ub[i])

    # Magic method associated with MutableSequence
    def __setitem__(self,
                    key: Key,
                    val: Union[DOF, Real]):
        if isinstance(key, Integral):
            i = key
        elif isinstance(key, str):  # Implement string indexing
            i, = np.where(self.names == key)
            if not i.size:
                raise IndexError("Element not found.")
        else:
            raise TypeError("Wrong type as index. Supply either an integer"
                            " or string for index.")

        if isinstance(val, DOF):
            self._x[i] = val.x
            self.names[i] = val.name
            self.free[i] = val.is_free()
            self._lb[i] = val.min
            self._ub[i] = val.max
        else:
            self._x[i] = float(val)

    # Magic method associated with MutableSequence
    def __delitem__(self, key: Key) -> None:
        """
        Delete a DOF
        Caution: Use this sparingly
        TODO:  Should we return a new DOFs object instead of deleting inplace?

        Args:
            key (string or Integer): Index

        Returns:
            None
        """
        if isinstance(key, Integral):
            i = key
        elif isinstance(key, str):
            i, = np.where(self.names == key)
            if not i.size:
                raise IndexError("Element not found.")
        else:
            raise TypeError("Wrong type as index. Supply either an integer"
                            " or string for index.")

        self._x = np.delete(self._x, i)
        self.names = np.delete(self.names, i)
        self.free = np.delete(self.free, i)
        self._lb = np.delete(self._lb, i)
        self._ub = np.delete(self._ub, i)

    # Magic method associated with MutableSequence
    def __len__(self) -> Integral:
        # TODO: Question - Shall we use self.x or self._x here?
        return len(self.free)

    # Magic method associated with MutableSequence
    def __contains__(self, item: Union[DOF, Real, str]):
        """
        Overloaded __contains__ method

        Args:
            item:

        Returns:

        """
        if isinstance(item, Real):
            return item in self._x
        elif isinstance(item, str):
            return item in self.names
        elif isinstance(item, DOF):
            if item.name in self.names:
                i, = np.where(self.names == item.name)
                return all([np.isclose(item.x, self._x[i]),
                            item.is_free() == self.free[i],
                            np.isclose(item.min, self._lb[i]),
                            np.isclose(item.max, self._ub[i])])
            else:
                return False
        else:
            raise TypeError(
                "Wrong type. Only str, DOF, or Real types accepted.")

    # Magic method associated with MutableSequence
    def __iter__(self):
        self.i = 0
        return self

    # Magic method associated with MutableSequence
    def __next__(self):
        # TODO: Question - Shall we use self.x or self._x here?
        i = self.i
        self.i += 1
        if i < len(self._x):
            return DOF(self.names[i], self._x[i], self.free[i],
                       self._lb[i], self._ub[i])
        else:
            raise StopIteration

    # Magic method associated with MutableSequence
    def __reversed__(self):
        for i in reversed(range(len(self._x))):
            yield DOF(self.names[i], self._x[i], self.free[i],
                      self._lb[i], self._ub[i])

    # Magic method associated with MutableSequence
    def __add__(self, other):
        """
        TODO: Bharat's comment: Shall we check for duplicates or blindly merge
        all of them. If we are blindly merging, we should provide a method
        to check for duplicate dofs in two DOFs objects

        For the time being, blindly extending the other dofs without any
        consideration for other DOFs

        Args:
            other:

        Returns:

        """
        names = np.append(self.names, other.names)
        x = np.append(self._x, other._x)
        free = np.append(self.free, other.free)
        lb = np.append(self._lb, other._lb)
        ub = np.append(self._ub, other._ub)
        return DOFs(x, names, free, lb, ub)

    # Method of MutableSequence
    def insert(self,
               i: Union[Integral, IntArray],
               val: Union[DOF, DOFs]) -> None:
        """
        Insert a new value into DOFs.

        Caution: This method is provided for consistency with
        MutableSequence. Use this sparingly or not at all

        Args:
            key: str or Integer
            val: New value to be inserted
        """

        if val.name is None:
            raise ValueError(
                "Can't assign name automatically under insertion operation.")
        if isinstance(val, DOF):
            self.names = np.insert(self.names, i, val.name)
        else:
            self.names = np.insert(self.names, val.names)

        self._x = np.insert(self._x, i, val.x)
        self.free = np.append(self.free, val.is_free())
        self._lb = np.append(self._lb, val.min)
        self._ub = np.append(self._ub, val.max)

    # Method of MutableSequence
    def append(self, val: Union[Real, DOF]) -> None:
        """
        Append a DOF to the list of DOFs

        Caution: This method is provided for consistency with
        MutableSequence. Use this sparingly or not at all.

        Args:
            val:

        Returns:

        """
        if isinstance(val, Real):
            self._x = self._x.append(val)
            self.names = "x[{}]".format(len(self._x) - 1)
            self.free = self.free.append(True)
            self._lb = self._lb.append(np.NINF)
            self._ub = self._ub.append(np.inf)
        else:
            self.names = self.names.append(
                "x[{}]".format(len(self._x)) if not val.name else val.name)
            self._x = self._x.append(val.x)
            self.free = self.free.append(val.is_free())
            self._lb = self._lb.append(val.min)
            self._ub = self._ub.append(val.max)

    # Method of MutableSequence
    def reverse(self) -> None:
        """
        Reverses the DOF sequence
        """
        self.names = np.flip(self.names)
        self._x = np.flip(self._x)
        self.free = np.flip(self.free)
        self._ub = np.flip(self._ub)
        self._lb = np.flip(self._lb)

    # Method of MutableSequence
    def extend(self, other) -> None:
        """
        Extends the DOFs object with dofs from another DOFs object

        TODO: Bharat's comment: Shall we check for duplicates or blindly merge
        all of them. If we are blindly merging, we should provide a method
        to check for duplicate dofs in two DOFs objects

        Args:
            other_dofs (DOFs): DOFs object

        Returns:

        """
        self.names = np.append(self.names, other.names)
        self._x = np.append(self._x, other._x)
        self.free = np.append(self.free, other.free)
        self._ub = np.append(self._ub, other._ub)
        self._lb = np.append(self._lb, other._lb)

    # Method of MutableSequence
    def pop(self, i=None):
        """
        Pops the DOF at specified position. If no position is given, last DOF
        is removed.
        Args:
            i:

        Returns:

        """
        i = i if i else -1
        x, self._x = self._x[i], np.delete(self._x, i)
        name, self.names = self.names[i], np.delete(self.names, i)
        free, self.free = self.free[i], np.delete(self.free, i)
        lb, self._lb = self._lb[i], np.delete(self._lb, i)
        ub, self._ub = self._ub[i], np.delete(self._ub, i)

        return DOF(name, x, free, lb, ub)

    # Method of MutableSequence
    def remove(self, value: DOF):
        for i, dof in enumerate(self):
            if dof == value:
                self.names = np.delete(self.names, i)
                self._x = np.delete(self._x, i)
                self.free = np.delete(self.free, i)
                self._lb = np.delete(self._lb, i)
                self._ub = np.delete(self._ub, i)

    def fix(self, key: Key) -> None:
        if isinstance(key, str):
            i, = np.where(self.names == key)
        else:
            i = key
        self.free[i] = False

    def unfix(self, key: Key) -> None:
        if isinstance(key, str):
            i, = np.where(self.names == key)
        else:
            i = key
        self.free[i] = True

    def fix_all(self) -> None:
        self.free = np.full(len(self._x), False)

    def unfix_all(self):
        """
        Make vall DOFs variable
        Caution: Make sure the bounds are well defined
        Returns:

        """
        self.free = np.full(len(self._x), True)

    def any_free(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if any free DOF is found
        """
        return np.any(self.free)

    def any_fixed(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if any fixed DOF is found
        """
        return not np.all(self.free)

    def all_free(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if all DOFs are free to vary
        """
        return np.all(self.free)

    def all_fixed(self) -> bool:
        """
        Checks for any free DOFs

        Returns:
            True if all DOFs are fixed
        """
        return not np.any(self.free)

    @property
    def x(self) -> RealArray:
        return self._x[self.free]

    @x.setter
    def x(self, x: RealArray) -> None:
        """

        Args:
            x: Array of values to set x
            Note: This method blindly broadcasts a single value.
            So don't supply a single value unless you really desire
        """
        self._x[self.free] = x

    @property
    def full_x(self) -> RealArray:
        """
        Return all x even the fixed ones

        Returns:
            Pruned DOFs object without any fixed DOFs
        """
        return self._x

    @property
    def lower_bounds(self) -> RealArray:
        """

        Returns:

        """
        return self._lb[self.free]

    @lower_bounds.setter
    def lower_bounds(self, lower_bounds):
        self._lb[self.free] = lower_bounds

    @property
    def upper_bounds(self):
        return self._ub[self.free]

    @upper_bounds.setter
    def upper_bounds(self, upper_bounds):
        self._ub[self.free] = upper_bounds

    @property
    def bounds(self):
        return (self.lower_bounds, self.upper_bounds)

    def update_upper_bound(self, key: Key, val: Real):
        if isinstance(key, str):
            self._ub[np.where(self.names == key)] = val
        else:
            self._ub[key] = val

    def update_lower_bound(self, key: Key, val: Real):
        if isinstance(key, str):
            self._lb[np.where(self.names == key)] = val
        else:
            self._lb[key] = val

    def update_bounds(self, key: Key, val: Tuple[Real]):
        if isinstance(key, str):
            self._lb[np.where(self.names == key)] = val[0]
            self._ub[np.where(self.names == key)] = val[1]
        else:
            self._lb[key] = val[0]
            self._ub[key] = val[1]

    # TODO: Move this method else where.
    @classmethod
    def from_function(cls, func):
        """
        Returns a list of DOFs from a function

        This class method returns the list of DOFs from a given optimizable
        function. The optimizable function is expected to be a method from
        sub-classes of Optimizable. If a generic function is given, an
        attempt is made to identify the DOFs in the function

        Args:
            func: Objective function for optimization

        Returns:
            List of DOFs (object of DOFs class)
        """
        # Convert user-supplied function-like things to actual functions:
        func = function_from_user(func)
        try:
            dofs = func.__self__.dofs
            return dofs
        except: # Decorate with optimizable and do further processing
            func = optimizable(func)

        # First, get a list of the objects and any objects they depend on:
        owners = get_owners(func.__self__)

        # Eliminate duplicates, preserving order:
        owners = unique(owners)

        dof_owners = []
        #indices = []
        fixed = []
        names = []

        # Store the dof_owners and indices for all dofs,
        # even if they are fixed. It turns out that this is the
        # information needed to convert the individual function
        # gradients into the global Jacobian.
        for owner in owners:
            ox = owner.get_dofs()
            ndofs = len(ox)
            # If 'fixed' is not present, assume all dofs are not fixed
            if hasattr(owner, 'fixed'):
                fixed += list(owner.fixed)
            else:
                fixed += [False] * ndofs

            for jdof in range(ndofs):
                dof_owners.append(owner)
                #indices.append(jdof)

            # Check for names:
            if hasattr(owner, 'names'):
                names += [str(owner) + '.' + name for name in owner.names]
            else:
                names += ['{}.x[{}]'.format(owner, k) for k in range(ndofs)]

        return DOFs(dof_owners, names, fixed)

    # TODO: Move this method else where.
    @classmethod
    def from_functions(cls, funcs):
        """
        Returns a list of variable (free) dofs from a list of optimizables

        This class method returns the list of free DOFs from a given list
        of optimizable functions. The optimizable functions are expected to
        be methods from sub-classes of Optimizable. If a generic function
        is present (not from Optimizable class), an attempt is made to
        identify dofs in the function. The dofs are pruned to remove fixed
        DOFs.

        Args:
            funcs: A sequence of callable functions.

        Returns:
            DOFs object containing free dofs
        """

        dofs = DOFs.from_function(funcs[0])
        for func in funcs[1:]:
            dofs += DOFs.from_function(func)

    @classmethod
    def from_functions_old(cls, funcs):

        # Don't do this here. Convert all user-supplied function-like things
        # to actual functions:
        #funcs = [function_from_user(f) for f in funcs]

        # First, get a list of the objects and any objects they depend on:
        all_owners = []
        for j in funcs:
            all_owners += get_owners(j.__self__)

        # Eliminate duplicates, preserving order:
        all_owners = unique(all_owners)

        # Go through the objects, looking for any non-fixed dofs:
        x = []
        dof_owners = []
        indices = []
        mins = []
        maxs = []
        names = []
        fixed_merged = []
        for owner in all_owners:
            ox = owner.get_dofs()
            ndofs = len(ox)
            # If 'fixed' is not present, assume all dofs are not fixed
            if hasattr(owner, 'fixed'):
                fixed = list(owner.fixed)
            else:
                fixed = [False] * ndofs
            fixed_merged += fixed

            # Check for bound constraints:
            if hasattr(owner, 'mins'):
                omins = owner.mins
            else:
                omins = np.full(ndofs, np.NINF)
            if hasattr(owner, 'maxs'):
                omaxs = owner.maxs
            else:
                omaxs = np.full(ndofs, np.Inf)

            # Check for names:
            if hasattr(owner, 'names'):
                onames = [name + ' of ' + str(owner) for name in owner.names]
            else:
                onames = ['x[{}] of {}'.format(k, owner) for k in range(ndofs)]

            for jdof in range(ndofs):
                if not fixed[jdof]:
                    x.append(ox[jdof])
                    dof_owners.append(owner)
                    indices.append(jdof)
                    names.append(onames[jdof])
                    mins.append(omins[jdof])
                    maxs.append(omaxs[jdof])

        # Now repeat the process we just went through, but for only a
        # single element of funcs. The results will be needed to
        # handle gradient information.
        func_dof_owners = []
        func_indices = []
        func_fixed = []
        # For the global dof's, we store the dof_owners and indices
        # only for non-fixed dofs. But for the dof's associated with
        # each func, we store the dof_owners and indices for all dofs,
        # even if they are fixed. It turns out that this is the
        # information needed to convert the individual function
        # gradients into the global Jacobian.
        for func in funcs:
            owners = get_owners(func.__self__)
            f_dof_owners = []
            f_indices = []
            f_fixed = []
            for owner in owners:
                ox = owner.get_dofs()
                ndofs = len(ox)
                # If 'fixed' is not present, assume all dofs are not fixed
                if hasattr(owner, 'fixed'):
                    fixed = list(owner.fixed)
                else:
                    fixed = [False] * ndofs
                f_fixed += fixed

                for jdof in range(ndofs):
                    f_dof_owners.append(owner)
                    f_indices.append(jdof)
                    # if not fixed[jdof]:
                    #    f_dof_owners.append(owner)
                    #    f_indices.append(jdof)
            func_dof_owners.append(f_dof_owners)
            func_indices.append(f_indices)
            func_fixed.append(f_fixed)

        # Check whether derivative information is available:
        grad_avail = True
        grad_funcs = []
        for func in funcs:
            # Check whether a gradient function exists:
            owner = func.__self__
            grad_func_name = 'd' + func.__name__
            if not hasattr(owner, grad_func_name):
                grad_avail = False
                break
            # If we get here, a gradient function exists.
            grad_funcs.append(getattr(owner, grad_func_name))

        dofs = DOFs(names, lower_bounds=mins, upper_bounds=maxs)

        # Make sure which ones are required and which ones are not
        dofs.funcs = funcs
        dofs.nfuncs = len(funcs)
        dofs.nparams = len(x)
        dofs.nvals = None  # We won't know this until the first function eval.
        dofs.nvals_per_func = np.full(dofs.nfuncs, 0)
        dofs.dof_owners = dof_owners
        dofs.indices = np.array(indices)
        # self.names = names
        # self.mins = np.array(mins)
        # self.maxs = np.array(maxs)
        dofs.all_owners = all_owners
        dofs.func_dof_owners = func_dof_owners
        dofs.func_indices = func_indices
        dofs.func_fixed = func_fixed
        dofs.grad_avail = grad_avail
        dofs.grad_funcs = grad_funcs

        return dofs

    # TODO: Move this method else where.
    def f(self, x=None):
        """
        Return the vector of function values. Result is a 1D numpy array.

        If the argument x is not supplied, the functions will be
        evaluated for the present state vector. If x is supplied, then
        first set_dofs() will be called for each object to set the
        global state vector to x.
        """
        if x is not None:
            self.x = x

        # Autodetect whether the functions return scalars or vectors.
        # For now let's do this on every function eval for
        # simplicity. Maybe there is some speed advantage to only
        # doing it the first time (if self.nvals is None.)
        val_list = []
        for j, func in enumerate(self.funcs):
            f = func()
            if isinstance(f, (np.ndarray, list, tuple)):
                self.nvals_per_func[j] = len(f)
                val_list.append(np.array(f))
            else:
                self.nvals_per_func[j] = 1
                val_list.append(np.array([f]))

        logger.debug('Detected nvals_per_func={}'.format(self.nvals_per_func))
        self.nvals = np.sum(self.nvals_per_func)
        return np.concatenate(val_list)

    # TODO: Move this method else where.
    def jac(self, x=None):
        """
        Return the Jacobian, i.e. the gradients of all the functions that
        were originally supplied to Dofs(). Result is a 2D numpy
        array.

        If the argument x is not supplied, the Jacobian will be
        evaluated for the present state vector. If x is supplied, then
        first set_dofs() will be called for each object to set the
        global state vector to x.
        """
        if not self.grad_avail:
            raise RuntimeError('Gradient information is not available for this Dofs()')

        if x is not None:
            self.x = x

        # grads = [np.array(f()) for f in self.grad_funcs]

        start_indices = np.full(self.nfuncs, 0)
        end_indices = np.full(self.nfuncs, 0)

        # First, evaluate all the gradient functions, and autodetect
        # how many rows there are in the gradient for each function.
        grads = []
        for j in range(self.nfuncs):
            grad = np.array(self.grad_funcs[j]())
            # Above, we cast to a np.array to be a bit forgiving in
            # case the user provides something other than a plain 1D
            # or 2D numpy array. Previously I also had flatten() for
            # working with Florian's simsgeo function; this may not
            # work now without the flatten.

            # Make sure grad is a 2D array (like the Jacobian)
            if grad.ndim == 1:
                # In this case, I should perhaps handle the rare case
                # of a function from R^1 -> R^n with n > 1.
                grad2D = grad.reshape((1, len(grad)))
            elif grad.ndim == 2:
                grad2D = grad
            else:
                raise ValueError('gradient should be 1D or 2D')

            grads.append(grad2D)
            this_nvals = grad2D.shape[0]
            if self.nvals_per_func[j] > 0:
                assert self.nvals_per_func[j] == this_nvals, \
                    "Number of rows in gradient is not consistent with number of entries in the function"
            else:
                self.nvals_per_func[j] = this_nvals

            if j > 0:
                start_indices[j] = end_indices[j - 1]
            end_indices[j] = start_indices[j] + this_nvals

        self.nvals = np.sum(self.nvals_per_func)

        results = np.zeros((self.nvals, self.nparams))
        # Loop over the rows of the Jacobian, i.e. over the functions
        # that were originally provided to Dofs():
        for jfunc in range(self.nfuncs):
            start_index = start_indices[jfunc]
            end_index = end_indices[jfunc]
            grad = grads[jfunc]

            # Match up the global dofs with the dofs for this particular gradient function:
            for jdof in range(self.nparams):
                for jgrad in range(len(self.func_indices[jfunc])):
                    # A global dof matches a dof for this function if the owners and indices both match:
                    if self.dof_owners[jdof] == self.func_dof_owners[jfunc][jgrad] and self.indices[jdof] == \
                            self.func_indices[jfunc][jgrad]:
                        results[start_index:end_index, jdof] = grad[:, jgrad]
                        # If we find a match, we can exit the innermost loop:
                        break

        # print('finite-difference Jacobian:')
        # fd_jac = self.fd_jac()
        # print(fd_jac)
        # print('analytic Jacobian:')
        # print(results)
        # print('difference:')
        # print(fd_jac - results)
        return results

    # TODO: Move this method else where.
    def fd_jac(self, x=None, eps=1e-7, centered=False):
        """
        Compute the finite-difference Jacobian of the functions with
        respect to all non-fixed degrees of freedom. Either a 1-sided
        or centered-difference approximation is used, with step size
        eps.

        If the argument x is not supplied, the Jacobian will be
        evaluated for the present state vector. If x is supplied, then
        first get_dofs() will be called for each object to set the
        global state vector to x.

        No parallelization is used here.
        """

        if x is not None:
            self.x = x

        logger.info('Beginning finite difference gradient calculation for functions ' + str(self.funcs))

        x0 = self.x
        logger.info('  nparams: {}, nfuncs: {}, nvals: {}'.format(self.nparams, self.nfuncs, self.nvals))
        logger.info('  x0: ' + str(x0))

        # Handle the rare case in which nparams==0, so the Jacobian
        # has size (nvals, 0):
        if self.nparams == 0:
            if self.nvals is None:
                # We don't know nvals yet. In this case, we could
                # either do a function eval to determine it, or
                # else return a 2d array of size (1,0), which is
                # probably the wrong size. For safety, let's do a
                # function eval to determine nvals.
                self.f()
            jac = np.zeros((self.nvals, self.nparams))
            return jac

        if centered:
            # Centered differences:
            jac = None
            for j in range(self.nparams):
                x = np.copy(x0)

                x[j] = x0[j] + eps
                self.x = x
                # fplus = np.array([f() for f in self.funcs])
                fplus = self.f()
                if jac is None:
                    # After the first function evaluation, we now know
                    # the size of the Jacobian.
                    jac = np.zeros((self.nvals, self.nparams))

                x[j] = x0[j] - eps
                self.x = x
                fminus = self.f()

                jac[:, j] = (fplus - fminus) / (2 * eps)

        else:
            # 1-sided differences
            f0 = self.f()
            jac = np.zeros((self.nvals, self.nparams))
            for j in range(self.nparams):
                x = np.copy(x0)
                x[j] = x0[j] + eps
                self.x = x
                fplus = self.f()

                jac[:, j] = (fplus - f0) / eps

        # Weird things may happen if we do not reset the state vector
        # to x0:
        self.x = x0
        return jac


class Optimizable(Callable, Hashable, metaclass=InstanceCounterABCMeta):
    """
    Callable ABC that provides useful features for optimizable objects.

    The class defines methods that are used by simsopt to know 
    degrees of freedoms (DOFs) associated with the optimizable
    objects. All derived objects have to implement method f that defines
    the objective function. However the users are not expected to call
    *f* to get the objective function. Instead the users should call
    the Optimizable object directly.

    Optimizable and its subclasses define the optimization problem. The
    optimization problem can be thought of a DAG, which each instance of
    Optimizable being a vertex in the DAG. Each Optimizable object can
    take other Optimizable objects as inputs and through this container
    logic, the edges of the DAG are defined.

    Alternatively, the input Optimizable objects can be thought as parents
    to the current Optimizable object. In this approach, the last grand-child
    defines the optimization problem by embodying all the elements of the
    parents and grand-parents. Each DOF defined in a parent gets passed down
    to the children. And each call to child instance gets in turn propagated
    to the parent.

    Currently, this back and forth propagation of DOFs and function calls
    happens at run time.

    Note: __init__ takes instances of subclasses of Optimizable as
          input and modifies them to define the children for input objects
    """
    # Bharat's comment: I think we should deprecate set_dofs and get_dofs
    # in favor of 'dof' or 'state' or 'x' property name? I think if we go
    # this route, having a function to collect dofs for any arbitrary function
    # or class is needed. For functions, it is straight-forward, all the
    # arguments to the function are DOFs unless indicated somehow. If the
    # arguments are Sequences or numpy array, each individual element is a
    # DOF. The bounds # can be supplied similar to scipy.least_squares. Same
    # is the case with objects of a class.
    #
    # For subclasses of Optimizable, instead of making set_dofs and
    # get_dofs as abstract method we will define a new abstract method called
    # collect_dofs to collect dofs into a DOFs object.

    def __init__(self,
                 x0: RealArray = None,
                 names: StrArray = None,
                 fixed: BoolArray = None,
                 lower_bounds: RealArray = None,
                 upper_bounds: RealArray = None,
                 funcs_in: Sequence[Optimizable] = None):
        """
        Args:
            x0: Initial state (or initial values of DOFs)
            names: Human identifiable names for the DOFs
            fixed: Array describing whether the DOFs are free or fixed
            lower_bounds: Lower bounds for the DOFs
            upper_bounds: Upper bounds for the DOFs
            funcs_in: Optimizable objects to define the optimization problem
                      in conjuction with the DOFs
        """
        self._dofs = DOFsDataFrame(
            x0, names, np.logical_not(fixed) if fixed is not None else None,
            lower_bounds, upper_bounds)

        # Generate unique and immutable representation for different
        # instances of same class
        self._id = ImmutableId(next(self.__class__._ids))
        self.name = self.__class__.__name__ + str(self._id.id)

        # Assign self as child to parents
        self.parents = funcs_in if funcs_in is not None else []
        for parent in self.parents:
            parent.add_child(self)

        # Obtain unique list of the ancestors
        self.ancestors = self.get_ancestors()

        # Compute the indices of all the DOFs
        dof_indices = [0]
        free_dof_size = 0
        full_dof_size = 0
        for opt in (self.ancestors + [self]):
            size = opt.local_free_dof_size
            free_dof_size += size
            full_dof_size += opt.local_dof_size
            dof_indices.append(free_dof_size)
        self.dof_indices = dict(zip(self.ancestors + [self],
                                    zip(dof_indices[:-1], dof_indices[1:])))
        self._free_dof_size = free_dof_size + self.local_free_dof_size
        self._full_dof_size =  full_dof_size + self.local_dof_size

        self._children = [] # This gets populated when the object is passed
                            # as argument to another Optimizable object
        self.new_x = True   # Set this True for dof setter and set it to False
                            # after evaluation of function if True

    def __str__(self):
        return self.name

    def __hash__(self) -> int:
        hash_str = hashlib.sha256(self.name.encode('utf-8')).hexdigest()
        return int(hash_str, 16) % 10**32  # 32 bit int as hash

    def __eq__(self, other: Optimizable) -> bool:
        """
        Checks the equality condition

        Args:
            other: Another object of subclass of Optimizable

        Returns: True only if both are the same objects.

        """
        #return (self.__class__ == other.__class__ and
        #        self._id.id == other._id.id)
        return self.name == other.name

    def __call__(self, x: RealArray = None):
        if x is not None:
            self.x = x
        if self.new_x:
            self._val = self.f()
            self.new_x = False
        return self._val

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        """
        Define this method in subclasses
        Args:
            *args:
            **kwargs:

        Returns:

        """

    def add_child(self, other: Optimizable) -> None:
        self._children.append(other)

    @property
    def free_dof_size(self) -> Integral:
        """
        Length of free DOFs associated with the Optimizable including those
        of parents
        """
        return self._free_dof_size

    @property
    def local_free_dof_size(self) -> Integral:
        return self._dofs.reduced_len

    @property
    def local_dof_size(self) -> Integral:
        return len(self._dofs)

    @property
    def dof_size(self) -> Integral:
        """
        Length of DOFs associated with the Optimizable including those
        of parents
        Returns:

        """
        return self._full_dof_size

    def _update_free_dof_size_indices(self) -> None:
        """
        Call the function to update the DOFs lengths for this instance and
        those of the children.

        Call whenever DOFs are fixed or unfixed. Recursively calls the
        function in children

        TODO: This is slow because it walks through the graph repeatedly
        TODO: Develop a faster scheme.
        TODO: Alternatively ask the user to call this manually from the end
        TODO: node after fixing/unfixing any DOF
        """
        dof_indices = [0]
        free_dof_size = 0
        for opt in (self.ancestors + [self]):
            size = opt.local_free_dof_size
            free_dof_size += size
            dof_indices.append(free_dof_size)
        self.dof_indices = dict(zip(self.ancestors + [self],
                                    zip(dof_indices[:-1], dof_indices[1:])))

        # Update the reduced length of children
        for child in self._children:
            child._update_free_dof_size_indices()

    @property
    def dofs(self) -> RealArray:
        return np.concatenate([opt._dofs.x for
                               opt in (self.ancestors + [self])])

    @dofs.setter
    def dofs(self, x: RealArray) -> None:
        for opt, indices in self.dof_indices.items():
            opt.local_dofs = x[indices[0]:indices[1]]
        for child in self._children:
            child._set_new_x()

    @property
    def local_dofs(self) -> RealArray:
        return self._dofs.x

    @local_dofs.setter
    def local_dofs(self, x: RealArray) -> None:
        self._dofs.x = x
        self.new_x = True

    @property
    def state(self) -> RealArray:
        return self.dofs

    @property
    def local_state(self) -> RealArray:
        return self.local_dofs

    @property
    def x(self) -> RealArray:
        return self.dofs

    @x.setter
    def x(self, x: RealArray):
        self.dofs = x

    @property
    def local_x(self):
        return self.local_dofs

    def _set_new_x(self):
        self.new_x = True
        for child in self._children:
            child._set_new_x()

    @property
    def bounds(self) -> Tuple[RealArray, RealArray]:
        return (self.lower_bounds, self.upper_bounds)

    @property
    def local_bounds(self) -> Tuple[RealArray, RealArray]:
        return self._dofs.bounds

    @property
    def lower_bounds(self) -> RealArray:
        opts = self.ancestors + [self]
        return np.concatenate([opt._dofs.lower_bounds for opt in opts])

    @property
    def local_lower_bounds(self) -> RealArray:
        return self._dofs.lower_bounds

    @property
    def upper_bounds(self) -> RealArray:
        opts = self.ancestors + [self]
        return np.concatenate([opt._dofs.upper_bounds for opt in opts])

    @property
    def local_upper_bounds(self) -> RealArray:
        return self._dofs.upper_bounds

    def get(self, key: Key) -> Real:
        """
        Return a the value of degree of freedom specified by its name or by index.
        """
        return self._dofs[key].x

    def set(self, key: Key, new_val: Real) -> None:
        """
        Set a degree of freedom specified by its name or by index.
        """
        self._dofs[key] = new_val

    def is_fixed(self, key: Key) -> bool:
        """
        Tells if the dof specified with its name or by index is fixed
        """
        return not self.is_free(key)

    def is_free(self, key: Key) -> bool:
        """
        Tells if the dof specified with its name or by index is fixed
        """
        if isinstance(key, str):
            return self._dofs.loc[key, 'free']
        else:
            return self._dofs.iloc[key, 'free']

    def fix(self, key: Key) -> None:
        """
        Set the fixed attribute for a given degree of freedom, specified by dof_str.

        TODO: Question: Should we use ifix similar to pandas' loc and iloc?
        """
        self._dofs.fix(key)
        self._update_free_dof_size_indices()

    def unfix(self, key: Key) -> None:
        """
        Set the fixed attribute for a given degree of freedom, specified by dof_str.
        """
        self._dofs.unfix(key)
        self._update_free_dof_size_indices()

    def fix_all(self) -> None:
        """
        Set the 'fixed' attribute for all degrees of freedom.
        """
        #self.dof_fixed = np.full(len(self.get_dofs()), True)
        self._dofs.fix_all()
        self._update_free_dof_size_indices()

    def unfix_all(self) -> None:
        """
        Set the 'fixed' attribute for all degrees of freedom.
        """
        #self.dof_fixed = np.full(len(self.get_dofs()), False)
        self._dofs.unfix_all()
        self._update_free_dof_size_indices()

    def get_ancestors(self) -> list[Optimizable]:
        ancestors = []
        for parent in self.parents:
            ancestors += parent.get_ancestors()
        ancestors += self.parents
        return list(dict.fromkeys(ancestors))


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

    def __call__(self, x):
        f = getattr(self.obj, self.attr)
        if callable(f):
            return f(x)
        else:
            return f

    @deprecated(version='0.0.2', reason="Call the object directly. Don't assume"
                                        " J method will be present.")
    def J(self):
        return getattr(self.obj, self.attr)

    #def dJ(self):
    #    return getattr(self.obj, 'd' + self.attr)

    # Bharat's comment: The following two needs to be better defined
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
