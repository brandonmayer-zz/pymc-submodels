'''
Created on Jun 9, 2011

@author: daniel
'''

import pymc as pm
import numpy as np
from  mixture_models import gaussian_mixture

def fixed_mixture_weights(alpha, K):
    pip = pm.Dirichlet('pip', theta = np.ones(K)*alpha)
    
    @pm.deterministic(dtype = float)
    def pi(value = np.ones(K)/K, pip = pip):
        val = np.hstack((pip, (1-np.sum(pip))))        
        return val
    
    return {'pip': pip, 'pi': pi}

def model(data, tau_like, mu_base, tau_base, alpha, K):    
    mdl_fixed_weights = fixed_mixture_weights(alpha, K)    
    mdl_gaussian = gaussian_mixture(data, mdl_fixed_weights['pi'], 
                     tau_like, mu_base, tau_base, K)
    
    return [mdl_fixed_weights, mdl_gaussian]