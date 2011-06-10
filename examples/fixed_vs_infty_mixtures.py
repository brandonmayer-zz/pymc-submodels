'''Test mixture finite and infinite mixture models'''

import pymc as pm
import numpy as np
import pylab as pl
from gaussian_mixtures import fixed_mixture_model as fmm
from gaussian_mixtures import infty_mixture_model as imm

def run_fixed_model(data, K):
    '''Return number of components used in a simple Gaussian mixture'''    
    tau_like = 0.1
    mu_base = 0.
    tau_base = 0.01
    alpha = 1.
    mdl = fmm.model(data, tau_like, mu_base, tau_base, alpha, K)
    mcmc = pm.MCMC(mdl)
    #mcmc.use_step_method(pm.AdaptiveMetropolis, mdl[0]['pip'])
    mcmc.sample(10000, 1000, 5, verbose = 0)
    
    return mcmc.trace('z')[:]

def run_infty_model(data, K = 20):
    '''Return number of components used in a simple Gaussian mixture'''    
    tau_like = 0.5
    mu_base = 0.
    tau_base = 0.01
    alpha = 1.
    mdl = imm.model(data, tau_like, mu_base, tau_base, alpha, K)    
    mcmc = pm.MCMC(mdl)
    #mcmc.use_step_method(pm.AdaptiveMetropolis, mdl[0]['pip'])
    
    mcmc.sample(10000, 1000, 5, verbose = 0)
    
    return mcmc.trace('z')[:]

def fixedmixture_model(data):
    '''Run mixture model with different K and see what is the distribution
    of mixture components usage'''
    z3 = run_fixed_model(data, 5)
    z10 = run_fixed_model(data, 30)
    
    pl.figure(1)
    pl.subplot(1, 2, 1)
    pl.hist([len(np.unique(z3[i,:])) for i in range(len(z3))])
    pl.xlabel('Components used')
    pl.ylabel('Frequency')    
    pl.title('K = 5')
    pl.xlim([0, 8])    
    pl.subplot(1, 2, 2)
    pl.hist([len(np.unique(z10[i,:])) for i in range(len(z10))])
    pl.xlabel('Components used')
    pl.ylabel('Frequency')    
    pl.title('K = 30')    
    pl.xlim([0, 8])
    pl.savefig('./plots/dist_components_fixedmixture.pdf')
    pl.close()
    
    print('Finished.')
    
def infitymixture_model(data):
    '''Run mixture model with different K and see what is the distribution
    of mixture components usage'''
    z10 = run_infty_model(data, 5)
    z20 = run_infty_model(data, 30)
    
    pl.figure(1)
    pl.subplot(1, 2, 1)
    pl.hist([len(np.unique(z10[i,:])) for i in range(len(z10))])
    pl.xlabel('Components used')
    pl.ylabel('Frequency')    
    pl.title('Truncation K = 5')   
    
    pl.xlim([0, 8])
    
    pl.subplot(1, 2, 2)
    pl.hist([len(np.unique(z20[i,:])) for i in range(len(z20))])    
    pl.xlabel('Components used')
    pl.ylabel('Frequency')
    pl.title('Truncation K = 30')    
    pl.xlim([0, 8])
    
    pl.savefig('./plots/dist_components_inftymixture.pdf')
    pl.close()
    print('Finished.')

if __name__ == '__main__':
    DATA = np.array([10., 11., 12., -10., -11., -12.])
    fixedmixture_model(DATA)
    infitymixture_model(DATA)