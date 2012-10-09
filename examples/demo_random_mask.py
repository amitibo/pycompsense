"""
Solve the Missing Pixels problem using two approaches:

   * Wavelet basis as a sparsifying basis
   * Total Vatiation
"""

from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import compsense


def solve_wavelet(plot_results=False):
    """
    Main Function
    """

    #
    # Generate environment
    #
    P = compsense.problems.probMissingPixels(wavelet_family='bior3.3', sigma=1e-3)
  
    #
    # Regularization parameter
    #
    tau = 0.05

    #
    # Solve an L1 recovery problem:
    # minimize  1/2|| Ax - b ||_2^2  +  tau ||x||_1
    #
    alg = compsense.algorithms.TwIST(
        P,
        tau,
        stop_criterion=1,
        tolA=1e-3
        )
    x = alg.solve()
    
    #
    # The solution x is the reconstructed signal in the sparsity basis.
    # Use the function handle P.reconstruct to use the coefficients in
    # x to reconstruct the original signal.
    #
    y  = P.reconstruct(x)

    if not plot_results:
        return
    
    #
    # Show results
    #
    plt.figure()
    plt.imshow(P.signal, cmap=cm.gray, origin='lower')
    plt.title('Original Image')

    plt.figure()
    plt.imshow(P.b.reshape(P.A.out_signal_shape), cmap=cm.gray, origin='lower')
    plt.title('Distorted Image')
    
    plt.figure()
    plt.imshow(y, cmap=cm.gray, origin='lower')
    plt.title('Reconstructed Image using Wavelet Basis')
    
    plt.figure()
    plt.semilogy(alg.times, alg.objectives, lw=2)
    plt.title('Evolution of the objective function (Wavelet Basis)')
    plt.xlabel('CPU time (sec)')
    plt.grid(True)
    
    plt.figure()
    plt.semilogy(alg.times, alg.mses, lw=2)
    plt.title('Evolution of the mse (Wavelet Basis)')
    plt.xlabel('CPU time (sec)')
    plt.grid(True)

    plt.show()

    
def solve_TV(plot_results=False):
    """
    Main Function
    """

    from skimage.filter import tv_denoise

    def Psi(x, threshold):
        """
        Deblurring operator.
        
        Arguments:
        ----------
        x : array-like, shape = [m, n]
            Estimated signal
           
        threshold : float
            Threshold for the deblurring algorithm
        """
    
        img_estimated = tv_denoise(x, weight=threshold/2, n_iter_max=4)
    
        return img_estimated
    
        
    def Phi(x):
        """
        Regularization operator.
        
        Arguments:
        ----------
        x : array-like, shape = [m, n]
            Input signal vector.
        """
    
        dy = np.zeros_like(x[:, :])
        dx = np.zeros_like(x[:, :])
        
        dy[:-1] = np.diff(x[:, :], axis=0)
        dx[:, :-1] = np.diff(x[:, :], axis=1)
        phi = np.sum(np.sqrt(dy**2 + dx**2)) 
            
        return phi

    #
    # Generate environment
    #
    P = compsense.problems.probMissingPixels(wavelet_family=None, sigma=1e-3)
  
    #
    # Regularization parameter
    #
    tau = 0.1
    tolA = 1e-8
    
    #
    # Solve an L1 recovery problem:
    # minimize  1/2|| Ax - b ||_2^2  +  tau || \nabla x ||_1
    #
    alg = compsense.algorithms.TwIST(
        P,
        tau,
        psi_function=Psi,
        phi_function=Phi,
        tolA=tolA
        )
    x = alg.solve()
   
    #
    # The solution x is the reconstructed signal in the sparsity basis.
    # Use the function handle P.reconstruct to use the coefficients in
    # x to reconstruct the original signal.
    #
    y  = P.reconstruct(x)

    if not plot_results:
        return
    
    #
    # Show results
    #
    plt.figure()
    plt.imshow(P.signal, cmap=cm.gray, origin='lower')
    plt.title('Original Image')

    plt.figure()
    plt.imshow(P.b.reshape(P.A.out_signal_shape), cmap=cm.gray, origin='lower')
    plt.title('Distorted Image')
    
    plt.figure()
    plt.imshow(y, cmap=cm.gray, origin='lower')
    plt.title('Reconstructed Image (Total Variation)')
    
    plt.figure()
    plt.semilogy(alg.times, alg.objectives, lw=2)
    plt.title('Evolution of the objective function (Total Variation)')
    plt.xlabel('CPU time (sec)')
    plt.grid(True)
    
    plt.figure()
    plt.semilogy(alg.times, alg.mses, lw=2)
    plt.title('Evolution of the mse (Total Variation)')
    plt.xlabel('CPU time (sec)')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    
    solve_wavelet(True)
    solve_TV(True)
