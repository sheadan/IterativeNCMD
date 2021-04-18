'''Iterative nonlinear chirp mode decomposition.

The work contained herein is adapted from:
Tu, Guowei, Xingjian Dong, Shiqian Chen, Baoxuan Zhao, Lan Hu, and Zhike Peng. “Iterative Nonlinear Chirp Mode Decomposition: A Hilbert-Huang Transform-like Method in Capturing Intra-Wave Modulations of Nonlinear Responses.” Journal of Sound and Vibration 485 (October 2020): 115571. https://doi.org/10.1016/j.jsv.2020.115571.
'''
import numpy as np
from scipy import linalg
from scipy.integrate import cumtrapz
from scipy.sparse import spdiags, diags, identity, csc_matrix, csr_matrix
from scipy.signal import welch
from scipy.sparse.linalg import inv as sparseinv
from scipy.sparse.linalg import spsolve


class INCMD:
    def __init__(self, signal, t, iters=1000, rho=0.5, mu=0.5, tol=1e-8, verbose=False):
        # Save the signal and time array
        self.signal = signal.flatten()
        self.t = t.flatten()
        # Parameters for optimization
        self.iters = iters
        self.rho = rho
        self.mu = mu
        self.tol = tol
        # Generate D matrix (modified second-order differences)
        self.D = self.get_D_matrix()
        # Compute matrix product D^T D, which is used for a number of computations
        self.DTD = csc_matrix(self.D.T @ self.D)
        # Compute the frequency-step filter and g-mode prefactor:
        n = self.D.shape[0]
        self.f_filter = sparseinv(((2/mu)*(self.DTD)+identity(n, format='csc')))
        self.g_factor = (1/rho)*(self.DTD)
        # Verbose attribute for printing
        self.verbose = verbose

    def get_D_matrix(self):
        # Get a D matrix that's sized n x n for a given signal of length n
        n = len(self.signal)
        superdiag = np.ones(n)
        diagonal = -2*np.ones(n)
        diagonal[-1] = -1
        diagonal[0] = -1
        D = spdiags(np.asarray([diagonal, superdiag, superdiag]), 
                    np.array([0, -1, 1]), 
                    n, n, format='csc')
        return D

    def NCMD(self):
        # Grab the signal and matrix D
        signal = self.signal
        t = self.t

        # Initial frequency estimate by welch method
        f_0 = self.welch_method_estimate()

        # Initialize phi matrices
        phi_1 = self.compute_new_phi_1(f_0)
        phi_2 = self.compute_new_phi_2(f_0)

        # Initialize estimates for gd1 and gd2
        g_d1 = self.compute_new_g(phi_1)
        g_d2 = self.compute_new_g(phi_2)

        # Initial g_0 computation
        g_0 = self.compute_mode(phi_1, g_d1, phi_2, g_d2)

        # Set the g_previous and f_previous variables:
        g_previous = g_0
        f_previous = f_0

        # Iterate and refine NCM
        for i in range(self.iters):
            if self.verbose: 
                print(i, end=" | ")

            # Update estimates for gd1 and gd2
            g_d1 = self.compute_new_g(phi_1)
            g_d2 = self.compute_new_g(phi_2)

            # Update the frequency estimate
            f_bar = self.compute_new_f(g_d1, g_d2, f_previous)

            # Initialize phi matrices
            phi_1 = self.compute_new_phi_1(f_bar)
            phi_2 = self.compute_new_phi_2(f_bar)

            # Compute the new mode g^i
            g = self.compute_mode(phi_1, g_d1, phi_2, g_d2)

            # Compute change in the mode definition
            loss = linalg.norm(g-g_previous, ord=2)**2
            loss /= linalg.norm(g_previous, ord=2)**2

            # If the change is less than tol, break
            if loss < self.tol:
                break

            # Update previous-iteration variables
            g_previous = g
            f_previous = f_bar

        if self.verbose:
            print("Executed {} iterations to solve for mode.".format(i))

        return f_bar, g, g_d1, g_d2

    def decompose(self, modes:int=1):
        # Take a copy of the original signal:
        original_signal = self.signal.copy()
        # Results lists
        f_bars = []
        gs = []
        g_d1s = []
        g_d2s = []
        # Compute NCM for each prescribed mode number:
        for i in range(modes):
            f_bar, g, g_d1, g_d2 = self.NCMD()
            # Save the results:
            f_bars.append(f_bar.copy())
            gs.append(g.copy())
            g_d1s.append(g_d1.copy())
            g_d2s.append(g_d2.copy())
            # Subtract the current mode from the data:
            self.signal -= g

        # Restore the original signal
        self.signal = original_signal

        # Return the results lists:
        return f_bars, gs, g_d1s, g_d2s

    def welch_method_estimate(self):
        # Grab signal and time vectors
        signal = self.signal
        t = self.t
        # Create an initial estimate on the frequency vector using the Welch PSD method
        nperseg = int(signal.shape[0]/16)
        nperseg += 0.5*nperseg
        f, Pxx = welch(signal, (1/(t[1]-t[0])), nperseg=nperseg)
        init_guess = f[np.argmax(Pxx)]
        return init_guess*np.ones(signal.shape)


    def compute_new_phi_1(self, new_f):
        # Compute new phi_1
        integrated_f = cumtrapz(new_f, self.t, initial=0)
        phi_diag = np.cos(2*np.pi*integrated_f)
        return diags(phi_diag)


    def compute_new_phi_2(self, new_f):
        # Compute new phi_2
        integrated_f = cumtrapz(new_f, self.t, initial=0)
        phi_diag = np.sin(2*np.pi*integrated_f)
        return diags(phi_diag)


    def compute_new_g(self, phi):
        def sparse_solve(A, b):
            return spsolve(A, b)
        # Compute the phi^T phi term (which, for diagonal phi is diag(phi) squared)
        phi_diagonal = phi.data[0,:]
        phi_square_diag = phi_diagonal*phi_diagonal
        # Compute the new mode
        new_G_A = (self.g_factor + diags(phi_square_diag, format='csc'))
        new_G_b = phi @ self.signal
        new_G = sparse_solve(new_G_A, new_G_b)
        #new_G = new_G.flatten() # Commented this out because flatten seemed to densify this sparse matrix (bad for memory)
        return new_G


    def compute_mode(self, phi_1, g_d1, phi_2, g_d2):
        # Compute and return the estimated signal from mode g_k
        #mode = np.dot(phi_1, g_d1) + np.dot(phi_2, g_d2)
        #mode = phi_1 @ g_d1 + phi_2 @ g_d2
        mode = phi_1.data[0,:] * g_d1 + phi_2.data[0,:] * g_d2
        return mode


    def compute_new_f(self, g1, g2, old_f):
        # Gather mu, D, and dt
        mu = self.mu
        D = self.D
        dt = self.t[1] - self.t[0]
        # Compute gradients
        dg1dt = np.gradient(g1, dt)
        dg2dt = np.gradient(g2, dt)
        #print(dg1dt.shape,dg2dt.shape, g1.shape, g2.shape)
        # Compute frequency step
        delta_f_num = (1/(2*np.pi))*((g2*dg1dt-g1*dg2dt))
        delta_f_denom = (g1**2+g2**2)
        delta_f = delta_f_num/delta_f_denom
        # And compute the frequency step vector
        step = self.f_filter @ delta_f
        #print(np.mean(prefactor), np.mean(delta_f))
        # Finally, compute the vector of frequency steps
        new_f = old_f.reshape(-1,1) + step.reshape(-1,1)
        new_f = np.asarray(new_f).flatten()
        #print("new f", new_f.shape, type(new_f))
        return new_f

