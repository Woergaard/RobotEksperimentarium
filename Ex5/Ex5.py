
### SIR RESAMPLING ###

import numpy as np
from scipy.stats import norm

# Define the Gaussian distributions
gaussians = [(0.3, 2.0, 1.0), (0.4, 5.0, 2.0), (0.3, 9.0, 1.0)]

def p(x, gaussian):
    '''
    Funktionen definerer en posefordeling.
    Argumenter:
        x:  antal datapunkter
    '''
    return sum(w * norm.pdf(x, mu, sigma) for w, mu, sigma in gaussians)

def q_uniform(x):
    '''
    Funktionen definerer en uniform proposalfordeling.
    Argumenter:
        x:  antal datapunkter
    '''
    return np.random.uniform(0, 15, size=int(x))

def q_gauss(x, drawSample : bool):
    '''
    Funktionen definerer en normaltfordelt proposalfordeling.
    Argumenter:
        x:  antal datapunkter
    '''
    if drawSample: 
        return np.random.normal(5, 4, size=int(x))
    else: 
        return norm.pdf(x, 5, 4)

def sir(k, distibrution, gaussian):
    '''
    Funktionen udfører SIR-algoritmen.
    Argumenter:
        k:  antal datapunkter / x fra før 
        distribution:  fordelingen, der skal resamples fra
    '''
    # Compute weights
    if distibrution == 'uniform':
        # Generate initial samples
        samples = q_uniform(k)
        weights = p(samples) / (1/15)#q_uniform(k) #This is wrong: You should divide by the probability density of the sample. For this uniform distribution it is 1/15.
    elif distibrution == 'gauss': 
        # Generate initial samples
        samples = q_gauss(k, True)
        weights = p(samples, gaussian) / q_gauss(samples, False) #This is again wrong: Here you should divide by the probability density of the samples 
                                                       #evaluated in the density function for the  Gaussian a.k.a Normal distribution.
    # Normalize weights
    weights /= sum(weights) 

    # Resample according to weights
    resamples = np.random.choice(samples, size=k, p=weights)

    return resamples
