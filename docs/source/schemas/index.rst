Configuration Objects
=====================

Data Model Containers
---------------------

This package contains high level analysis functions which take in data sets (S-parameter data,
data collected from RF sweep measurements, etc) and operate on them with lower level functions following
some mathematical algorithm. To facilitate both python scripting, and interacting the package through a
user interface like a GUI or CLI, the package defines configuration objects called a ``DataModelContainer``.
These can be supplied either pointers to a data set in a file, or the in memory representation
of the file (like an RMEMeas object). The ``DataModelContainer`` can invoke the ``load()`` function
which will return the preffered in-memory representation of whatever dataset is being requested by the function -
provided a supported input was provided.


For example, when scripting in python a function called ``fn`` may request a parameter called ``my_s11`` that is expected to be
a ``S11`` data model container. This could be supplied as an in memory representation that was already loaded previously
in the script, in this example called ``data``.

.. code-block:: python
    
    fn(my_s11 = data)


Importantly, in the case of a UX the value for ``my_s11`` can be specified as path to the data set. In
this example  to a `.dut` file, a text format produced by NIST's S-parameter calibration service.

.. code-block:: python

    fn(my_s11 = 'path/to/my/file.dut')

It could also be represented in the rmellipse RMEMeas format, by providing a path to an HDF5 file. If the dataset is nested
in an HDF5 group, continue the file path to the group.

.. code-block:: python

    fn(my_s11 = 'path/to/my/file.hdf5/group/inside/file')


You could also provide a DataModelContainer itself.

.. code-block:: python

    fn(my_s11 = microcalorimetry.configs.S11('path/to/my/file.hdf5/group/inside/file'))


Inside the analysis function, the parameter ``my_s11`` can be cast into the correct container.
The ``.load()`` function is used to load in and get the in-memory representation. The data set can be freed
 and the confguration object
preserves a pointer to the data set or file path that can be loaded in again later. If a pre-loaded
dataset is provided to the function, then python garbage collector will keep the data set in memory since
``my_s11`` preserves a reference to it. 

.. code-block:: python

    # load in as needed
    s11_data  = microcalorimetry.configs.S11(my_s11).load()
    # do something with s11_data
    new_value = some_other_operation(s11_data)
    # free up memory
    del s11_data
    # re load if needed again
    s11_data = my_s11.load()

This gives analysis scripts the ability
to easily load and free memory as needed. It also gives
a python script invoking the analysis function the ability to
force the data set to stay in memory by pre-loading it, saving load time
at the cost of increased memory usage.

Serial Dictionaries
-------------------

Some measurement and analysis functions will request
a dictionary object with a particular structure, which are inherited
from a base class called ``Serial Dictionary``. Fields in the dictionary contain measurement settings,
or analysis settings, or a ``DataModelContainer``.
They can be supplied by giving either a manually created dictionary in a python script,

.. code-block:: python

    my_config = {'key': 'value', ... }

Or can be written in a file using a standard markdown format (YAML,  JSON, or ExperimentParameters).

.. code-block:: yaml

    key: 'value'

Then represented in a script as a path which points to the actual
configuration.

.. code-block:: python

    my_config = 'my_config.yml'


In either case, either the file path or a dictionary constructed in a python
script can be passed into a function requesting a type of Serial Dictionary.

.. code-block:: python

    my_config = microcalorimetery.configs.MyConfigType(my_config)

When cast, as Serial Dictionary is validated against a schema to validate
the structure. They are defined using `json schema <https://json-schema.org/>`__.

This facilitates interoperability of analysis and measurmeent functions with Python
scripting environments, and with user interfaces where a dictionary may need
to be written into a file. It also provides some validation to
complex configuration files.

Configuration Reference
-----------------------

This section includes detailed schema descriptions of every type of confguration object used in this package.
This documentation is auto-generated based on the schema-definitions. Objects in this reference may also correspond
to Python objects that can be imported in the ``microcalorimetry.configs`` submodule. See the API reference for
more information on those objects.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   data-models.rst
   measurements/index.rst
   analysis-schemas.rst

