# IterativeNCMD
Iterative Nonlinear Chirp Mode Decomposition

The code in this repository aims to create a working copy of the Iterative Nonlinear Chirp Mode Decomposition technique proposed in Tu et. al in [1].

Note: In the paper, the authors utilize the Ljung-Box Q-test to test the signal after each chirp mode identification to determine if additional modes were needed. In this instantiation of the code, the user specifies the number of modes **a priori**. The decompose method implements the iterative nature of INCMD; if desired, I recommend you implement the Ljung-Box test in this method.

[1] Tu, Guowei, Xingjian Dong, Shiqian Chen, Baoxuan Zhao, Lan Hu, and Zhike Peng. “Iterative Nonlinear Chirp Mode Decomposition: A Hilbert-Huang Transform-like Method in Capturing Intra-Wave Modulations of Nonlinear Responses.” Journal of Sound and Vibration 485 (October 2020): 115571. [https://doi.org/10.1016/j.jsv.2020.115571.](https://doi.org/10.1016/j.jsv.2020.115571)
