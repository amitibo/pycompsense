"""

"""

from __future__ import division
import pkg_resources
import types    


def getResourcePath(name):
    """return the path to resource"""

    return pkg_resources.resource_filename(__name__, "data/%s" % name)
    

def isFunction(f):
    """
    Check if object is a function.
    """

    return isinstance(f, types.FunctionType) or isinstance(f, types.MethodType) or hasattr(f, '__call__')
    