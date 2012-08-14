"""

"""

from __future__ import division
import pkg_resources
import numpy as np
import types    


def getResourcePath(name):
    """return the path to resource"""

    return pkg_resources.resource_filename(__name__, "data/%s" % name)
    

def isFunction(f):
    """
    Check if object is a function.
    """

    return isinstance(f, types.FunctionType) or isinstance(f, types.MethodType) or hasattr(f, '__call__')

def softThreshold(x, threshold):
    """
    Apply Soft Thresholding
    
    Parameters
    ----------
    
    x : array-like
        Vector to which the soft thresholding is applied
        
    threshold : float
        Threhold of the soft thresholding
    
    Returns:
    --------
    y : array
        Result of the applying soft thresholding to x.
        
        .. math::
              
            y = sign(x) \star \max(\abs(x)-threshold, 0)
    """
    
    #
    # y = sign(x).*max(abs(x)-tau,0);
    #
    y = np.abs(x) - threshold
    y[y<0] = 0
    y[x<0] = -y[x<0]
    
    return y

    