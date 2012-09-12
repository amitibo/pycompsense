"""
Operators relating to the sparse problems.
"""

from __future__ import division
import numpy as np
import numpy.fft as npfft
import rwt
from rwt import wavelets


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

    def __init__(self, name, shape, in_signal_shape=None, out_signal_shape=None):

        if in_signal_shape==None:
            in_signal_shape = (shape[1], 1)
            out_signal_shape = (shape[0], 1)            
        elif out_signal_shape==None:
            if shape[0]==shape[1]:
                out_signal_shape = in_signal_shape
            else:
                out_signal_shape = (shape[0], 1)
            
        assert np.prod(in_signal_shape)==shape[1], 'Input signal shape does not conform to the shape of the operator'
        assert np.prod(out_signal_shape)==shape[0], 'Output signal shape does not conform to the shape of the operator'
            
        self._name = name
        self._shape = shape
        self._in_signal_shape = in_signal_shape
        self._out_signal_shape = out_signal_shape
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
    def in_signal_shape(self):
        """The shape of the input signal for the operator."""
        if self._conj:
            return self._out_signal_shape
        else:
            return self._in_signal_shape
    
    @property
    def out_signal_shape(self):
        """The shape of the output signal for the operator."""
        if self._conj:
            return self._in_signal_shape
        else:
            return self._out_signal_shape
        
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
    
    def _apply(self, x):
        """Apply the operator on the input signal. Should be overwritten by the operator."""
        
        raise NotImplementedError()
        
    def __call__(self, x):
        
        x = x.reshape((-1, 1))
        
        self._checkDimensions(x)

        return self._apply(x).reshape(self.out_signal_shape)
    
    
class opBlur(opBase):
    """
    Two-dimensional blurring operator. creates an blurring operator
    for M by N images. This function is used for the GPSR-based test
    problems and is based on the implementation by Figueiredo, Nowak 
    and Wright, 2007.

    Parameters
    ----------
    shape : (int, int)
        Shape of target images.

    """

    def __init__(self, shape):
        
        assert len(shape) == 2, "opBlur supports operations on 2D matrices only"
        m, n = shape
        size = m * n
        
        super(opBlur, self).__init__(
            name='Blur',
            shape=(size, size),
            in_signal_shape=shape
        )
        
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
        
    def _apply(self, x):

        if not self._conj:
            h = self._h
        else:
            h = self._h.conj()

        y = npfft.ifft2(h * npfft.fft2(x.reshape(self._in_signal_shape))).reshape((-1, 1))

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
    
    def __init__(self, shape, family='Daubechies', filter=8, levels=5, type='min'):

        assert len(shape) == 2, "opWavelet supports operations on 2D matrices only"
        size = shape[0] * shape[1]
        
        super(opWavelet, self).__init__(
            name='Wavelet',
            shape=(size, size),
            in_signal_shape=shape
        )
        
        family = family.lower()

        if family == 'daubechies':
            self._wavelet = wavelets.daubcqf(filter)[0]
        elif family == 'daubechies':
            self._wavelet = rwt.daubcqf(0)[0]
        else:
            raise Exception('Wavelet family %s is unknown' % family)

        self._level = levels
        
    def _apply(self, x):
        
        if self._conj:
            wf = rwt.mdwt
        else:
            wf = rwt.midwt 
            
        if np.isrealobj(x):
            y, l = wf(x.reshape(self._in_signal_shape), self._wavelet, self._level)
        else:
            [y1, l] = wf(x.real.reshape(self._in_signal_shape), self._wavelet, self._level)
            [y2, l] = wf(x.imag.reshape(self._in_signal_shape), self._wavelet, self._level)
            y = y1 + 1j*y2
             
        y.shape = (-1, 1)
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

    def _apply(self, x):
        
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

        super(opFoG, self).__init__(
            name='FoG',
            shape=(m, n),
            in_signal_shape=operators_list[-1].in_signal_shape,
            out_signal_shape=operators_list[0].out_signal_shape
        )
        self._operators_list = operators_list

    def _apply(self, x):

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
        
        in_signal_shape = operator.in_signal_shape
        if in_signal_shape[1] == 1:
            in_signal_shape = (in_signal_shape[0]*dim3, 1)
        else:
            in_signal_shape = (in_signal_shape[0], in_signal_shape[1], dim3)
            
        out_signal_shape = operator.out_signal_shape
        if out_signal_shape[1] == 1:
            out_signal_shape = (out_signal_shape[0]*dim3, 1)
        else:
            out_signal_shape = (out_signal_shape[0], out_signal_shape[1], dim3)
            
        super(op3DStack, self).__init__(
            name='3DStack',
            shape=(m*dim3, n*dim3),
            in_signal_shape=in_signal_shape,
            out_signal_shape=out_signal_shape
        )
        self._operator = operator
        self._dim3 = dim3

    def _apply(self, x):

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