'''
@author: daniel
'''

import pymc as pm
import numpy as np

def sticks(alpha, K):
    '''Creates a truncated stick-breaking construction of GEM distribution.
    Concentration parameter `alpha` and length K.
    AdaptiveMetropolis should be used over `pip`
    '''
    pip = pm.Beta('pip', alpha = 1., beta = alpha, size = K - 1)     
    
    @pm.deterministic(dtype = float)
    def pi(value = np.ones(K)/K, pip = pip):
        pip2 = np.hstack((pip.copy(), [1.]))        
        val = [pip2[k]*np.prod(1-pip2[0:k]) for k in range(K)]
        return val
    
    return {'pip': pip, 'pi': pi}
    
if __name__ == '__main__':
    dp_sticks(1., 100)