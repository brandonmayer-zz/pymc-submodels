'''
Created on Jun 9, 2011

@author: daniel
'''

import pymc as pm

def gaussian_mixture(data, pi, tau_like, mu_base, tau_base, K):
    '''
    Create mixture model for data using a Gaussian with mean mu_base 
    and precision tau_base as base, with K mixture components and
    alpha as a proxy for how many mixture components are expected.
    Observations with known precision tau_like. 
    pi are mixture weights
    '''
    n = len(data)
    # Component to which each data point belongs
    z = pm.Categorical('z', p = pi, size = n)
    
    # Parameters of each component    
    mu_k = pm.Normal('mu_k', mu = mu_base, tau = tau_base, size = K)
    
    # Observation model
    x = pm.Normal('x', mu = mu_k[z], tau = tau_like, value = data, 
                    observed = True)
    
    return {'z': z, 'mu_k': mu_k, 'x': x}