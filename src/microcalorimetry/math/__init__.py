"""
This module contains lower level math routines.

They deal with in-memory representation's of data only, and are utilized by the analysis model.
For example, a function in this module should specify exactly what type of object it expects,
and never refer to a DataModelContainer which could point to an arbitraty data-type.

Module are seperate into submodules based on the expected application of their use. Each method
should be an implementation of some mathematical process that can be described in a functional way.

"""
