"""
problems
========

A set of problems for testing and benchmarking algorithms for sparse
signal reconstruction. problemBase should be subclassed for creating
new problems.

..
    This module is based on MATLAB SPARCO Toolbox.
    Copyright 2008, Ewout van den Berg and Michael P. Friedlander
    http://www.cs.ubc.ca/labs/scl/sparco

.. codeauthor:: Amit Aides <amitibo@tx.technion.ac.il>

"""

from __future__ import division
import numpy as np
from .operators import *
from .utilities import *


class problemBase(object):
    """
    Base class for all CS problems. The problems follow
    the quation below:
    
    .. math::
    
        A x = b
        A = M B
    
    where :math:`A` is an operator acting on a sparse signal :math:`x`
    and :math:`b` is the observation vector. :math:`A` can be factored
    into :math:`M` which represents the system response and :math:`B`
    basis that sparsifies the signal.
    
    Attributes
    ----------
    name : string
        Name of problem.
    A : Instance of a subclass of opBase
        The :math:`A` matrix of the problem.
    M : Instance of a subclass of opBase
        :math:`M`, sampling matrix of the problem.
    B : Instance of a subclass of opBase
        :math:`B`, sparsifying basis matrix of the problem.
    b : array of arbitrary shape
        :math:'b', observation array.
    signal : array of arbitrary shape
        Signal in original basis (Not in the sparsifying basis)
    signal_shape : tuple of integers
        :math:'b', Shape of the signal in the sparsifying basis.
        
    Methods
    -------
    reconstruct : Reconstruct signal from sparse coefficients.
    """

    def __init__(self, name, noseed=False):
        """
        Parameters
        ----------
        name : str
            Name of the problem
        noseed: Boolean, optional (default=False)
            When False, the random seed is reset to 0.
        """

        self._name = name
        
        #
        # Initialize random number generators
        #
        if not noseed:
            np.random.seed(seed=0)

    @property
    def name(self):
        """Name of the problem
        """
        return self._name
        
    @property
    def A(self):
        """Response of the problem
        """
        return self._A
        
    @property
    def M(self):
        """Sampling matrix
        """
        return self._M
        
    @property
    def B(self):
        """Base matrix
        """
        return self._B
        
    @property
    def b(self):
        """Observation vector
        """
        return self._b
        
    @property
    def signal(self):
        """Signal (Not in sparsifying basis)
        """
        return self._signal
        
    @property
    def x0(self):
        """Solution to problem
        """
        return self._x0
        
    @property
    def signal_shape(self):
        """Shape of the signal
        """
        return self._signal_shape
        
    def _completeOps(self):
        """Finalize the reconstruction of the problem. Should be called by the constructor.
        """

        if not hasattr(self, '_M') and not hasattr(self, '_B'):
            raise Exception('At least one of the operator fileds _M or _B is required.')

        #
        # Define measurement matrix
        #
        if not hasattr(self, '_M'):
            self._M = opDirac(self._B.shape[0])
            operators = []
        else:
            operators = [self._M]
            
        #
        # Define sparsitry bases
        #
        if not hasattr(self, '_B'):
            self._B = opDirac(self._M.shape[1])
        else:
            operators.append(self._B)

        #
        # Define operator A if needed
        #
        if not hasattr(self, '_A'):
            if len(operators) > 1:
                self._A = opFoG(operators)
            else:
                self._A = operators[0]

        #
        # Define empty solution if needed
        #
        if not hasattr(self, '_x0'):
            self._x0 = []

        #
        # Get the size of the desired signal
        #
        if not hasattr(self, '_signal_shape'):
            if not hasattr(self, '_signal'):
                raise Exception('At least one of the fields _signal or _signal_shape is required.')
            self._signal_shape = self._signal.shape

    def reconstruct(self, x):
        """Reconstruct signal from sparse coefficients"""
        
        y = self._B(x).reshape(self._signal_shape)

        return y
    

class prob701(problemBase):
    """
    GPSR example: Daubechies basis, blurred Photographer.
    prob701 creates a problem structure.  The generated signal will
    consist of the 256 by 256 grayscale 'photographer' image. The
    signal is blurred by convolution with an 9 by 9 blurring mask and
    normally distributed noise with standard deviation SIGMA = 0.0055
    is added to the final signal.

    Examples
    --------
    >>> P = prob701()   # Creates the default 701 problem.

    References
    ----------

    [FiguNowaWrig:2007] M. Figueiredo, R. Nowak and S.J. Wright,
      Gradient projection for sparse reconstruction: Application to
      compressed sensing and other inverse problems, Submitted,
      2007. See also http://www.lx.it.pt/~mtf/GPSR

    """

    def __init__(self, sigma=np.sqrt(2)/256, noseed=False):
        """
        Parameters
        ----------
        sigma : float, optional (default=sqrt(2)/256)
            Standard deviation of the additive noise.
        noseed : bool, optional (default=False)
             When True, the nitialization of the random number
             generators is suppressed
        """
        super(prob701, self).__init__(name='blurrycam', noseed=noseed)

        #
        # Parse parameters and set problem name
        #
        self._sigma = sigma

        #
        # Set up the data
        #
        import matplotlib.pyplot as plt        
        signal = plt.imread(getResourcePath("/prob701_Camera.tif"))
        m, n = signal.shape

        #
        # Set up the problem
        #
        self._signal = signal.astype(np.float) / 256
        self._M = opBlur(signal.shape)
        self._B = opWavelet(signal.shape, 'Daubechies', 2)
        self._b = self._M(self._signal.reshape((-1, 1)))
        self._b += sigma * np.random.randn(m, n)
        self._x0 = self._B.T(self._signal)
        
        #
        # Finish up creation of the problem
        #
        self._completeOps()
        

def main():
    """
    Main Function
    """

    pass


if __name__ == '__main__':
    main()