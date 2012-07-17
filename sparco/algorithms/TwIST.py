from __future__ import division
import numpy as np
from ..utils import *

def _softThreshold(x, threshold):
    """
    Soft Thresholding
    """
    
    #
    # y = sign(x).*max(abs(x)-tau,0);
    #
    y = np.abs(x) - threshold
    y[y<0] = 0
    y[x<0] = -y[x<0]
    
    return y


def TwIST(
        y,
        A,
        tau,
        psi_function=_softThreshold,
        phi_function=None,
        lam1=1e-4,
        alpha=0,
        beta=0,
        AT=None,
        stop_criterion=1,
        tolA=0.01,
        debias=0,
        tolD=0.001,
        maxiter=1000,
        miniter=5,
        maxiter_debias=200,
        miniter_debias=5,
        init=0,
        enforce_monotone=True,
        sparse=True,
        true_x=None,
        verbose=True
        ):
    """
    This function solves the regularization problem 
    
        arg min_x = 0.5*|| y - A x ||_2^2 + tau phi( x ), 
    
    where A is a generic matrix and phi(.) is a regularizarion 
    function  such that the solution of the denoising problem 
    
        Psi_tau(y) = arg min_x = 0.5*|| y - x ||_2^2 + tau \phi( x ), 
    
    is known. 
    
    For further details about the TwIST algorithm, see the paper:
    
    J. Bioucas-Dias and M. Figueiredo, "A New TwIST: Two-Step
    Iterative Shrinkage/Thresholding Algorithms for Image 
    Restoration",  IEEE Transactions on Image processing, 2007.
    
    and
    
    J. Bioucas-Dias and M. Figueiredo, "A Monotonic Two-Step 
    Algorithm for Compressive Sensing and Other Ill-Posed 
    Inverse Problems", submitted, 2007.
    
    Authors: Jose Bioucas-Dias and Mario Figueiredo, October, 2007.
    
    Please check for the latest version of the code and papers at
    www.lx.it.pt/~bioucas/TwIST
    
    -----------------------------------------------------------------------
    Copyright (2007): Jose Bioucas-Dias and Mario Figueiredo
    
    TwIST is distributed under the terms of 
    the GNU General Public License 2.0.
    
    Permission to use, copy, modify, and distribute this software for
    any purpose without fee is hereby granted, provided that this entire
    notice is included in all copies of any software which is or includes
    a copy or modification of this software and in all copies of the
    supporting documentation for such software.
    This software is being provided "as is", without any express or
    implied warranty.  In particular, the authors do not make any
    representation or warranty of any kind concerning the merchantability
    of this software or its fitness for any particular purpose."
    ----------------------------------------------------------------------
    
     ===== Required inputs =============
    
     y: 1D vector or 2D array (image) of observations
        
     A: if y and x are both 1D vectors, A can be a 
        k*n (where k is the size of y and n the size of x)
        matrix or a handle to a function that computes
        products of the form A*v, for some vector v.
        In any other case (if y and/or x are 2D arrays), 
        A has to be passed as a handle to a function which computes 
        products of the form A*x; another handle to a function 
        AT which computes products of the form A'*x is also required 
        in this case. The size of x is determined as the size
        of the result of applying AT.
    
     tau: regularization parameter, usually a non-negative real 
          parameter of the objective  function (see above). 
     
    
     ===== Optional inputs =============
     
     'psi_function' = denoising function handle; handle to denoising function
             Default = soft threshold.
    
     'phi_function' = function handle to regularizer needed to compute the objective
             function.
             Default = ||x||_1
    
     'lam1' = lam1 parameters of the  TwIST algorithm:
              Optimal choice: lam1 = min eigenvalue of A'*A.
              If min eigenvalue of A'*A == 0, or unknwon,  
              set lam1 to a value much smaller than 1.
    
              Rule of Thumb: 
                  lam1=1e-4 for severyly ill-conditioned problems
                  lam1=1e-2 for mildly  ill-conditioned problems
                  lam1=1    for A unitary direct operators
    
              Default: lam1 = 0.04.
    
              Important Note: If (max eigenvalue of A'*A) > 1,
              the algorithm may diverge. This is  be avoided 
              by taking one of the follwoing  measures:
    
                 1) Set 'Monontone' = 1 (default)
                   
                 2) Solve the equivalenve minimization problem
    
              min_x = 0.5*|| (y/c) - (A/c) x ||_2^2 + (tau/c^2) \phi( x ), 
    
              where c > 0 ensures that  max eigenvalue of (A'A/c^2) <= 1.
    
     'alpha' = parameter alpha of TwIST (see ex. (22) of the paper)         
              Default alpha = alpha(lamN=1, lam1)
      
     'beta' = parameter beta of twist (see ex. (23) of the paper)
              Default beta = beta(lamN=1, lam1)            
    
     'AT'    = function handle for the function that implements
               the multiplication by the conjugate of A, when A
               is a function handle. 
               If A is an array, AT is ignored.
    
     'stop_criterion' = type of stopping criterion to use
                       0 = algorithm stops when the relative 
                           change in the number of non-zero 
                           components of the estimate falls 
                           below 'ToleranceA'
                       1 = stop when the relative 
                           change in the objective function 
                           falls below 'ToleranceA'
                       2 = stop when the relative norm of the difference between 
                           two consecutive estimates falls below toleranceA
                       3 = stop when the objective function 
                           becomes equal or less than toleranceA.
                       Default = 1.
    
     'tolA' = stopping threshold; Default = 0.01
    
     'debias'     = debiasing option: 1 = yes, 0 = no.
                    Default = 0.
                    
                    Note: Debiasing is an operation aimed at the 
                    computing the solution of the LS problem 
    
                            arg min_x = 0.5*|| y - A' x' ||_2^2 
    
                    where A' is the  submatrix of A obatained by
                    deleting the columns of A corresponding of components
                    of x set to zero by the TwIST algorithm
                    
    
     'tolD' = stopping threshold for the debiasing phase:
                    Default = 0.0001.
                    If no debiasing takes place, this parameter,
                    if present, is ignored.
    
     'maxiter' = maximum number of iterations allowed in the
                  main phase of the algorithm.
                  Default = 1000
    
     'miniter' = minimum number of iterations performed in the
                  main phase of the algorithm.
                  Default = 5
    
     'maxiter_debias' = maximum number of iterations allowed in the
                  debising phase of the algorithm.
                  Default = 200
    
     'miniter_debias' = minimum number of iterations to perform in the
                  debiasing phase of the algorithm.
                  Default = 5
    
     'init' must be one of {0,1,2,array}
                  0 -> Initialization at zero. 
                  1 -> Random initialization.
                  2 -> initialization with A'*y.
                  array -> initialization provided by the user.
                  Default = 0;
    
     'enforce_monotone' = enforce monotonic decrease in f. 
                  True -> enforce monotonicity
                  False -> don't enforce monotonicity.
                  Default = True;
    
     'sparse'   = Accelarates the convergence rate when the regularizer 
                  Phi(x) is sparse inducing, such as ||x||_1.
                  Default = True
                  
     'true_x' = if the true underlying x is passed in 
                   this argument, MSE evolution is computed
    
     'Verbose'  = work silently (0) or verbosely (1)
    
    ===================================================  
    ============ Outputs ==============================
      x = solution of the main algorithm
    
      x_debias = solution after the debiasing phase;
                     if no debiasing phase took place, this
                     variable is empty, x_debias = [].
    
      objective = sequence of values of the objective function
    
      times = CPU time after each iteration
    
      debias_start = iteration number at which the debiasing 
                     phase started. If no debiasing took place,
                     this variable is returned as zero.
    
      mses = sequence of MSE values, with respect to True_x,
             if it was given; if it was not given, mses is empty,
             mses = [].
    
      max_svd = inverse of the scaling factor, determined by TwIST,
                applied to the direct operator (A/max_svd) such that
                every IST step is increasing.
    ========================================================
    """
    
    compute_mse = 0
    plot_ISNR = 0
    phi_l1 = 0
    psi_ok = 0
    lamN = 1
    
    # 
    # constants and internal variables
    #
    for_ever = 1

    #
    # maj_max_sv: majorizer for the maximum singular value of operator A
    #
    max_svd = 1

    #
    # Set the defaults for outputs that may not be computed
    #
    debias_start = 0
    x_debias = np.array([])
    mses = np.array([])
    
    #
    # twist parameters
    #                
    rho0 = (1-lam1/lamN) / (1+lam1/lamN)
    
    if alpha == 0:
        alpha = 2 / (1 + np.sqrt(1-rho0**2.))
    
    if beta == 0:
        beta = alpha*2/(lam1+lamN)
    
    if stop_criterion not in range(4):
        raise Exception('Unknwon stopping criterion')
    
    #
    # if A is a function handle, we have to check presence of AT
    #
    if isFunction(A) and not isFunction(AT):
        raise Exception('The function handle for transpose of A is missing')
    
    #
    # if A is a matrix, we find out dimensions of y and x,
    # and create function handles for multiplication by A and A',
    # so that the code below doesn't have to distinguish between
    # the handle/not-handle cases
    #
    if not isFunction(A):
        AT = lambda x: np.zeros(y.shape)
        A = lambda x: np.dot(A, x.reshape((-1, 1))).reshape(y.shape)
    
    #
    # from this point down, A and AT are always function handles.
    # Precompute A'*y since it'll be used a lot
    Aty = AT(y)

    #
    #
    #
    if phi_function == None:
        phi_function = lambda x: np.sum(np.abs(x))
        phi_l1 = 1
        
    #--------------------------------------------------------------
    # Initialization
    #--------------------------------------------------------------
    if np.isscalar(init):
        if init == 0:
            #
            # initialize at zero, using AT to find the size of x
            #
            x = AT(np.zeros(y.shape))
        elif init == 1:
            #
            # initialize randomly, using AT to find the size of x
            #
            x = np.random.randn(AT(np.zeros(y.shape)).shape)
        elif init == 2:
            #
            # initialize x0 = A'*y
            #
            x = Aty
        else:
            raise Exception("Unknown 'Initialization' option")
    else:
        if A(init).shape == y.shape:
            x = init
        else:
            raise Exception("Size of initial x is not compatible with A")

    #
    # now check if tau is an array; if it is, it has to have the same size as x
    #
    if not np.isscalar(tau) and tau.shape != x.shape:
        raise Exception('Parameter tau has wrong dimensions; it should be scalar or size(x)')

    #
    # if the true x was given, check its size
    #
    #if np.logical_and(compute_mse, matcompat.size(true) != matcompat.size(x)):
    #    matcompat.error(np.array(np.hstack(('Initial x has incompatible size'))))
    
    
    #
    # if tau is large enough, in the case of phi = l1, thus psi = soft,
    # the optimal solution is the zero vector
    #
    if phi_l1:
        max_tau = np.max(np.abs(Aty))
        
        if tau >= max_tau and psi_ok == 0:
            x = np.zeros(Aty.shape)
            objective = [0.5*np.sum(y * y)]
            times = [0]
            
            # if compute_mse:
            #     mses[0] = np.sum((true.flatten(1)**2.))
            
            return x, x_debias, objective, debias_start, max_svd

    #
    # define the indicator vector or matrix of nonzeros in x
    #
    nz_x = x != 0
    num_nz_x = np.sum(nz_x)
    
    #
    # Compute and store initial value of the objective function
    #
    resid = y - A(x)
    prev_f = 0.5*np.sum(resid * resid) + tau*phi_function(x)

    #
    #% start the clock
    # t0 = cputime
    # times[0] = cputime-t0
    objective = [prev_f]
    
    # if compute_mse:
    #     mses[0] = np.sum(np.sum(((x-true)**2.)))
    
    cont_outer = 1
    iter = 1
    if verbose:
        print '\nInitial objective = %10.6e,  nonzeros=%7d' % (prev_f, num_nz_x)
    
    #
    # variables controling first and second order iterations
    #
    IST_iters = 0
    TwIST_iters = 0
    
    #
    # initialize
    #
    xm2 = x
    xm1 = x

    #--------------------------------------------------------------
    # TwIST iterations
    #--------------------------------------------------------------
    while cont_outer:
        #
        # gradient
        #
        grad = AT(resid)
        while True:
            #
            # IST estimate
            #
            x = psi_function(xm1 + grad/max_svd, tau/max_svd)
            if (IST_iters >= 2) or (TwIST_iters != 0):
                #
                # set to zero the past when the present is zero
                # suitable for sparse inducing priors
                #
                if sparse:
                    mask = x != 0
                    xm1 = xm1 * mask
                    xm2 = xm2 * mask
                
                #
                # two-step iteration
                #
                xm2 = (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x

                #
                # compute residual
                #
                resid = y - A(xm2)
                f = 0.5*np.sum(resid *resid) + tau*phi_function(xm2)
                
                if (f > prev_f) and enforce_monotone:
                    #
                    # do a IST iteration if monotonocity fails
                    #
                    TwIST_iters = 0
                else:
                    #
                    # TwIST iterations
                    #
                    TwIST_iters += 1
                    IST_iters = 0
                    x = xm2
                    
                    if TwIST_iters % 10000 == 0:
                        max_svd *= 0.9
                    #
                    # break while loop
                    #
                    break

            else:
                resid = y-A(x)
                f = 0.5*np.sum(resid * resid) + tau*phi_function(x)
                         
                if f > prev_f:
                    #
                    # if monotonicity  fails here  is  because
                    # max eig (A'A) > 1. Thus, we increase our guess
                    # of max_svs
                    #
                    max_svd *= 2
                    
                    if verbose:
                        print 'Incrementing S=%2.2e' % max_svd
                        
                    IST_iters = 0
                    TwIST_iters = 0
                    
                else:
                    TwIST_iters += 1
                    
                    #
                    # break while loop
                    #
                    break

        xm2 = xm1
        xm1 = x
        
        #
        # Update the number of nonzero components and its variation
        #
        nz_x_prev = nz_x
        nz_x = x != 0
        num_nz_x = np.sum(nz_x)
        num_changes_active = np.sum(nz_x != nz_x_prev)

        #
        # take no less than miniter and no more than maxiter iterations
        #
        if stop_criterion == 0:
            #
            # compute the stopping criterion based on the change
            # of the number of non-zero components of the estimate
            #
            criterion = num_changes_active
        elif stop_criterion == 1:
            #
            # compute the stopping criterion based on the relative
            # variation of the objective function.
            #
            criterion = np.abs(f-prev_f)/prev_f
        elif stop_criterion == 2:
            #
            # compute the stopping criterion based on the relative
            # variation of the estimate.
            #
            criterion = np.linalg.norm((x-xm1).ravel()) / np.linalg.norm(x.ravel())
        elif stop_criterion == 3:
            #
            # continue if not yet reached target value tolA
            #
            criterion = f;
        else:
            raise Exception('Unknwon stopping criterion');

        cont_outer = (iter <= maxiter) and (criterion > tolA)
        
        if iter <= miniter:
            cont_outer = 1

        iter += iter
        prev_f = f
        objective.append(f)
        
        # times(iter) = cputime-t0;

        # if compute_mse
        #     err = true - x;
        #     mses(iter) = (err(:)'*err(:));
        # end

        # % print out the various stopping criteria
        # if verbose
        #     if plot_ISNR
        #         fprintf(1,'Iteration=%4d, ISNR=%4.5e  objective=%9.5e, nz=%7d, criterion=%7.3e\n',...
        #             iter, 10*log10(sum((y(:)-true(:)).^2)/sum((x(:)-true(:)).^2) ), ...
        #             f, num_nz_x, criterion/tolA);
        #     else
        #         fprintf(1,'Iteration=%4d, objective=%9.5e, nz=%7d,  criterion=%7.3e\n',...
        #             iter, f, num_nz_x, criterion/tolA);
        #     end
        # end
            
    #
    #--------------------------------------------------------------
    # end of the main loop
    #--------------------------------------------------------------
    # Printout results
    #
    if verbose:
        print '\nFinished the main algorithm!\nResults:'
        print '||A x - y ||_2 = %10.3e' % np.sum(resid * resid)
        print '||x||_1 = %10.3e' % np.sum(np.abs(x))
        print 'Objective function = %10.3e' % f
        print 'Number of non-zero components = %d' % num_nz_x
        # print 'CPU time so far = %10.3e' % times[int(iter)-1]
    
    
    # #%--------------------------------------------------------------
    # #% If the 'Debias' option is set to 1, we try to
    # #% remove the bias from the l1 penalty, by applying CG to the
    # #% least-squares problem obtained by omitting the l1 term
    # #% and fixing the zero coefficients at zero.
    # #%--------------------------------------------------------------
    # if debias:
    #     if verbose:
    #         fprintf(1., '\n')
    #         fprintf(1., 'Starting the debiasing phase...\n\n')
        
        
    #     x_debias = x
    #     zeroind = x_debias != 0.
    #     cont_debias_cg = 1.
    #     debias_start = iter
    #     #% calculate initial residual
    #     resid = A[int(x_debias)-1]
    #     resid = resid-y
    #     resid_prev = np.dot(finfo(float).eps, np.ones(matcompat.size(resid)))
    #     rvec = AT[int(resid)-1]
    #     #% mask out the zeros
    #     rvec = rvec*zeroind
    #     rTr_cg = np.dot(rvec.flatten(0).conj(), rvec.flatten(1))
    #     #% set convergence threshold for the residual || RW x_debias - y ||_2
    #     tol_debias = np.dot(tolD, np.dot(rvec.flatten(0).conj(), rvec.flatten(1)))
    #     #% initialize pvec
    #     pvec = -rvec
    #     #% main loop
    #     while cont_debias_cg:
    #         #% calculate A*p = Wt * Rt * R * W * pvec
            
    #     if verbose:
    #         fprintf(1., '\nFinished the debiasing phase!\nResults:\n')
    #         fprintf(1., '||A x - y ||_2 = %10.3e\n', np.dot(resid.flatten(0).conj(), resid.flatten(1)))
    #         fprintf(1., '||x||_1 = %10.3e\n', np.sum(np.abs(x.flatten(1))))
    #         fprintf(1., 'Objective function = %10.3e\n', f)
    #         nz = x_debias != 0.0
    #         fprintf(1., 'Number of non-zero components = %d\n', np.sum(nz.flatten(1)))
    #         fprintf(1., 'CPU time so far = %10.3e\n', times[int(iter)-1])
    #         fprintf(1., '\n')

    # if compute_mse:
    #     mses = matdiv(mses, length(true.flatten(1)))
    
    return x, x_debias, objective, debias_start, max_svd

