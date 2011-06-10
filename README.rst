**************************************************
pymc-submodels: Common Bayesian submodels for PyMC
**************************************************

:Author: Daniel E. Acuna <acuna@principiapredicitva.com> Copyright (c) 2010-2011
:URL: http://blog.principiapredictiva.com  


Description
===========

Common submodels used in Bayesian modeling, implemented in PyMC.

Contents
========

1. Truncated GEM distribution (sticks) for Dirichlet Process
2. Gaussian finite and infinite mixtures (based on truncated GEM distribution)

Example
=======

Infinite mixture::

   import numpy as np
   import pymc as pm
   from pymc_submodels import infty_mixture_model as imm
   
   data = np.array([10., 11., 12., -10., -11., -12.])
   # Precision of Gaussian likelihood
   tau_like = 0.5
   # Mean and precision of Gaussian base distribution
   mu_base = 0. 
   tau_base = 0.01
   
   # Concentration parameter of Dirichlet process
   alpha = 1. 
   
   # Construct model using pymc_submodels package
   mdl = imm.model(data, tau_like, mu_base, tau_base, alpha)
   
   # Call PyMC
   mcmc = pm.MCMC(mdl)
   # Sample
   mcmc.sample(10000, 1000, 2)   
   # Stochastic 'z' component membership of each data point
   print np.mean(mcmc.trace('z')[:], 0)