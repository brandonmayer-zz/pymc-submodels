'''
Infinite mixture model based on Dirichlet process

@author: daniel
'''
from pymc_submodels import dp
from pymc_submodels import dp
from mixture_models import gaussian_mixture

def model(data, tau_like, mu_base, tau_base, alpha, K = 20):
    
    mdl_infty_weights = dp.sticks(alpha, K)
    
    mdl_gaussian = gaussian_mixture(data, mdl_infty_weights['pi'], 
                     tau_like, mu_base, tau_base, K)
    
    return [mdl_infty_weights, mdl_gaussian]