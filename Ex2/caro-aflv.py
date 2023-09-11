################################################
# 3.1 
################################################
"""
Your task here is to implement a program that performs sampling from a probability distribution using the Sampling-Importance-Resampling (SIR) algorithm, 
as described at the lectures and in the book section 4.3, particularly the section 4.3.2 on importance sampling. In this sub-exercise, we will therefore focus 
on the sampling procedure required for one time step in the particle filer applied to a time sequence, but will leave the actual implementation of the 
particle filter algorithm for a later exercise.
Consider the toy example that our robot is moving in an environment with only 1 dimension. 
The robot pose, in this case only position, can is this case be described by a stochastic scalar variable X. We will assume that the distribution of possible robot 
positions can be described by a mixture of 3 normal distributions (also known as a mixture of Gaussian model) with the probability density
p(x) = 0.3·N(x;2.0,1.0) + 0.4·N(x;5.0,2.0) + 0.3·N(x;9.0,1.0). N is the gaussian probability density. 

In the SIR algorithm we need a proposal distribution q(x) to generate initial samples from. Next we compute weights using the above stated function p(x). 
Since we are only implementing one time step of the particle filter, it is particularly important how we choose the proposal distribution q(x) and we will investigate two choices.

For this question we choose the proposal distribution q(x) to be a uniform distribution on the interval [0 : 15]. You can generate samples from such a distribution using 
for instance Python’s random pa- ckage.
You have to write a python program that produces a set of resampled robot poses x using the Sampling- Importance-Resampling algorithm and the above stated pose distribution p(x) 
and proposal distribution q(x). Show the distribution of samples after the resampling step for k = 20, 100, 1000 samples / particles. Plot a histogram of the samples 
together with the wanted pose distribution p(x) (Hint: Take care the the histogram should be scaled as a probability density function to be comparable with p(x)). 
How well does the histogram of samples fit with p(x) for the different choices of k? Can you imagine any problems occurring when using a uniform proposal distribution 
with our particular choice of p(x)?

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


#Definition af gaussian
gaussian = [(0.3, 2.0, 1.0), (0.4, 5.0, 2.0), (0.3, 9.0, 1.0)] 

# state pose distribution PDF (Probability Density Function)
def p(X) :
        return sum(w * norm.pdf(x, mu, sigma) for w, mu, sigma in gaussian)


# Proposal distribution, som tager intervalet [0 : 15] og vi har variablen x, som skal bruge til at resamble 
def q(x) :
        return np.random.uniform (0, 15, size=int(x) )
    
    

# Implementation af SIR algoritmen. 
def sir(k): 
    
    #Initial sample 
    sample = q(k)
    
    # Deraf udregner vi dens vægt 
    weight = p(sample)
    
    # normalization af vægten, hvad sker der specifikt 
    weight /= sum(weight)
    
    # resample i forhold til vægt 
    resample = np.random.choice(sample, size=k, p=weight)
    
    return resample 


# udførelse af SIR med de forskellige k værdier. 
for k in [20, 100, 1000]:
    resample = sir(k)
    
    # histogrammet 
  # histogrammet, bins =30, så vi får halve intervaller, da vi har fra 0-15. 
    counts, bins = np.histogram(resample, bins=30)
    
    # Normalize counts by total number of samples
    fractions = counts / k

    # Plot histogram of fractions
    plt.bar(bins[:-1], fractions, width=np.diff(bins), alpha=0.5, label=f'k={k}')

plt.legend() # A legend is a way to identify the different lines or curves in a plot.
plt.title('Sampling Importance Resampling Histogram')
plt.savefig("Caro forsøg")
plt.show()
plt.close