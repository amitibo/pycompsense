"""
Orthogonal wavelet based image deburring using TwIST:
Setting: cameraman, blur uniform 9*9.

This demo illustrates the computation of  

    xe = arg min 0.5 ||Ax-y||^2 + tau ||x||_1
            x

with the TwIST algorithm, where the operator A= H o W is the  
composition of the blur and DWT synthesis operator.

For further details about the TwIST algorithm, see the paper:

J. Bioucas-Dias and M. Figueiredo, "A New TwIST: Two-Step
Iterative Shrinkage/Thresholding Algorithms for Image 
Restoration",  IEEE Transactions on Image processing, 2007.

(available at   http://www.lx.it.pt/~bioucas/publications.html)


Authors: Jose Bioucas-Dias and Mario Figueiredo, 
Instituto Superior Tecnico, October, 2007
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sparco


def main():
    """
    Main Function
    """

    #
    # Generate environment
    #
    P = sparco.prob701(sigma=1e-3)
  
    yorig = P.signal
    b  = P.b
  
    m, n = P.A.shape
  
    #
    # Regularization parameter
    #
    tau = 0.00005

    #
    # Solve an L1 recovery problem:
    # minimize  1/2|| Ax - b ||_2^2  +  tau ||x||_1
    #
    x, dummy, obj, dummy, dummy = sparco.TwIST(
        P.b,
        P.A,
        tau,
        AT=P.A.T,
        true_x=P.B.T(P.signal.reshape((-1, 1))),
        stop_criterion=1,
        tolA=1e-3
        )
    


  
# % The solution x is the reconstructed signal in the sparsity basis. 
# % Use the function handle P.reconstruct to use the coefficients in
# % x to reconstruct the original signal.
#   y     = P.reconstruct(x);    % Use x to reconstruct the signal.

  
#   h_fig=figure(1);
#   set(h_fig,'Units','characters',...
#         'Position',[0 35 100 30]);
#   imagesc(yorig)
#   title('Original image');
#   colormap gray;
 

#   h_fig=figure(2);
#   set(h_fig,'Units','characters',...
#         'Position',[100 35 100 30]);
#   imagesc(reshape(b,P.signalSize))
#   title('Blurred and noisy image');
#   colormap gray;

#   h_fig=figure(3);
#   set(h_fig,'Units','characters',...
#         'Position',[0 0 100 30]);
#   imagesc(y)
#   title('Reconstructed image');
#   colormap gray;


# % plot the evolution of the objective function
#   h_fig=figure(4);
#   set(h_fig,'Units','characters',...
#         'Position',[100  0 100 30]);

#   semilogy(time,obj,'LineWidth',2)
#   title('Evolution of the objective function');  
#   xlabel('CPU time (sec)');
#   grid on
 
# % plot the evolution of the mse
#   h_fig=figure(5);
#   set(h_fig,'Units','characters',...
#         'Position',[200  0 100 30]);

#   semilogy(time,mse,'LineWidth',2)
#   title('Evolution of the mse');  
#   xlabel('CPU time (sec)');
#   grid on


if __name__ == '__main__':
    main()