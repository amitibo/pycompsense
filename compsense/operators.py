"""
Operators relating to the sparse problems.
"""

from __future__ import division
import numpy as np
import numpy.fft as npfft


class opBase(object):
    """
    Base class for operators

    Attributes
    ----------
    name : string
        Name of operator.
    shape : tuple
        The shape of the target for the operator.
    
    Methods
    -------
    """

    def __init__(self, name, shape, signal_shape=None):

        if signal_shape==None:
            signal_shape = (shape[1], 1)
            
        self._name = name
        self._shape = shape
        self._signal_shape = signal_shape
        self._conj = False
            
    @property
    def name(self):
        """Name of operator."""
        return self._name
        
    @property
    def shape(self):
        """The shape of the operator."""
        if self._conj:
            return self._shape[::-1]
        else:
            return self._shape
        
    @property
    def signal_shape(self):
        """The shape of the input signal for the operator."""
        return self._signal_shape
        
    @property
    def T(self):
        import copy

        new_copy = copy.copy(self)
        new_copy._conj = True
        return new_copy

    def _checkDimensions(self, x):

        if x.shape == (1, 1) and self._shape != (1, 1):
            raise Exception('Operator-scalar multiplication not yet supported')

        if x.shape[0] != self.shape[1]:
            raise Exception('Incompatible dimensions')

        if x.shape[1] != 1:
            raise Exception('Operator-matrix multiplication not yet supported')
            

class opBlur(opBase):
    """opBlur   Two-dimensional blurring operator

    opBlur(m, n) creates an blurring operator for M by N
    images. This function is used for the GPSR-based test problems
    and is based on the implementation by Figueiredo, Nowak and
    Wright, 2007.

    Copyright 2008, Ewout van den Berg and Michael P. Friedlander
    http://www.cs.ubc.ca/labs/scl/sparco
    """

    def __init__(self, m, n):
        
        super(opBlur, self).__init__(name='Blur', shape=(m*n, m*n), signal_shape=(m, n))
        
        yc = int(m/2 + 1)
        xc = int(n/2 + 1)

        #
        # Create blurring mask
        #
        h  = np.zeros((m, n))
        g = np.arange(-4, 5)
        for i in g:
            h[i+yc, g+xc] = 1 / (1 + i*i + g**2)

        h = npfft.fftshift(h)
        h /= h.sum()
        self._h = npfft.fft2(h)
        
    def __call__(self, x):

        self._checkDimensions(x)

        if not self._conj:
            h = self._h
        else:
            h = self._h.conj()

        y = npfft.ifft2(h * npfft.fft2(x.reshape(self._signal_shape))).reshape((-1, 1))

        if np.isrealobj(x):
            y = np.real(y)

        return y
        

class opWavelet(opBase):
    """opWavelet  Wavelet operator

    opwavelet(m, n, family, filter, levels, type) creates a wavelet
    operator of given FAMILY, for M by N matrices. The wavelet
    transformation is computed using the Rice Wavelet Toolbox.

    The remaining three parameters are optional. FILTER = 8
    specifies the filter length and must be even. LEVELS = 5 gives
    the number of levels in the transformation. Both M and N must
    be divisible by 2^LEVELS. TYPE = 'min' indicates what type of
    solution is desired; 'min' for minimum phase, 'max' for
    maximum phase, and 'mid' for mid-phase solutions.
    
    Copyright 2008, Rayan Saab, Ewout van den Berg and Michael P. Friedlander
    http://www.cs.ubc.ca/labs/scl/sparco
    """
    
    def __init__(self, m, n, family='Daubechies', filter=8, levels=5, type='min'):

        super(opWavelet, self).__init__(name='Wavelet', shape=(m*n, m*n), signal_shape=(m, n))
        
        family = family.lower()

        if family == 'daubechies':
            self._wavelet = 'db%d' % int(filter/2)
        elif family == 'daubechies':
            self._wavelet = 'haar'
        else:
            raise Exception('Wavelet family %s is unknown' % family)

        self._level = levels
        
        #
        # Create a reusable reconstruction tree
        #
        import pywt
        self._wp = pywt.WaveletPacket2D(
            data=np.ones(self._signal_shape),
            wavelet=self._wavelet,
            maxlevel=self._level,
            mode='per'
        )
        self._leaf_nodes = self._wp.get_leaf_nodes(decompose=True)
        
    def __call__(self, x):
        
        import pywt

        self._checkDimensions(x)

        if self._conj:
            wp = pywt.WaveletPacket2D(
                data=x.reshape(self._signal_shape),
                wavelet=self._wavelet,
                maxlevel=self._level,
                mode='per'
            )
            coeff = [n.data for n in wp.get_leaf_nodes(decompose=True)]
            y = np.array(coeff).reshape((-1, 1))
        else:
            coeff = x.reshape([len(self._leaf_nodes)] + list(self._leaf_nodes[0].data.shape))
            for i, node in enumerate(self._leaf_nodes):
                node.data = coeff[i]
            y = self._wp.reconstruct(update=False).reshape((-1, 1))

        return y


class opDirac(opBase):
    """
    Identity operator

    opDirac(N) creates the identity operator for vectors of length N. 

    Copyright 2008, Ewout van den Berg and Michael P. Friedlander
    http://www.cs.ubc.ca/labs/scl/sparco
    """

    def __init__(self, n):

        super(opDirac, self).__init__(name='Dirac', shape=(n, n))

    def __call__(self, x):
        
        self._checkDimensions(x)

        return x.copy()


class opFoG(opBase):
    """
    Concatenate a sequence of operators into a single operator.

    opFoG((OP1,OP2,...OPn)) creates an operator that successively
    applies each of the operators OP1, OP2, ..., OPn on a given
    input vector. In non-adjoint mode this is done in reverse
    order, starting with OPn.

    See also opDictionary

    Copyright 2008, Ewout van den Berg and Michael P. Friedlander
    http://www.cs.ubc.ca/labs/scl/sparco
    """

    def __init__(self, operators_list):
        
        if len(operators_list) == 0:
            raise Exception('At least one operator must be specified')

        #
        # Check operator consistency and space
        #
        m, n = operators_list[0].shape

        for oper in operators_list[1:]:
            m_, n_ = oper.shape

            if m_ != n:
                raise Exception('Operator %s is not consistent with the previous operators' % oper.name)

            n = n_

        super(opFoG, self).__init__(name='FoG', shape=(m, n))
        self._operators_list = operators_list

    def __call__(self, x):

        self._checkDimensions(x)

        if self._conj:
            y = self._operators_list[0].T(x)
            for oper in self._operators_list[1:]:
                y = oper.T(y)
        else:
            y = self._operators_list[-1](x)
            for oper in self._operators_list[-2::-1]:
                y = oper(y)

        return y


class op3DStack(opBase):
    """
    Extend an operator to process a stack of signals.
    """

    def __init__(self, operator, dim3):
        
        if not isinstance(operator, opBase):
            raise Exception('operator should be an instance of opBase.')

        #
        # Check operator consistency and space
        #
        m, n = operator.shape

        super(op3DStack, self).__init__(name='3DStack', shape=(m*dim3, n*dim3))
        self._operator = operator
        self._dim3 = dim3

    def __call__(self, x):

        self._checkDimensions(x)

        if self._conj:
            op = self._operator.T
        else:
            op = self._operator
        
        y = []
        for x_ in np.split(x, self._dim3):
            y.append(op(x_))
            
        return np.vstack(y)


def main():
    """
    Main Function
    """

    pass


if __name__ == '__main__':
    main()