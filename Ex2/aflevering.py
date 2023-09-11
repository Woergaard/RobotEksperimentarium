################################################
# 3.1 
################################################
"""
Your task here is to implement a program that performs sampling from a probability distribution using the Sampling-Importance-Resampling (SIR) algorithm, as described at the lectures and in the book section 4.3, particularly the section 4.3.2 on importance sampling. In this sub-exercise, we will there- fore focus on the sampling procedure required for one time step in the particle filer applied to a time sequence, but will leave the actual implementation of the particle filter algorithm for a later exercise.
Consider the toy example that our robot is moving in an environment with only 1 dimension. The robot pose, in this case only position, can is this case be described by a stochastic scalar variable X. We will assume that the distribution of possible robot positions can be described by a mixture of 3 normal distributions (also known as a mixture of Gaussian model) with the probability density
p(x) = 0.3·N(x;2.0,1.0) + 0.4·N(x;5.0,2.0) + 0.3·N(x;9.0,1.0). N is the gaussian probability density. 

In the SIR algorithm we need a proposal distribution q(x) to generate initial samples from. Next we compute weights using the above stated function p(x). Since we are only implementing one time step of the particle filter, it is particularly important how we choose the proposal distribution q(x) and we will investigate two choices.

For this question we choose the proposal distribution q(x) to be a uniform distribution on the interval [0 : 15]. You can generate samples from such a distribution using for instance Python’s random pa- ckage.
You have to write a python program that produces a set of resampled robot poses x using the Sampling- Importance-Resampling algorithm and the above stated pose distribution p(x) and proposal distribu- tion q(x). Show the distribution of samples after the resampling step for k = 20, 100, 1000 samples / particles. Plot a histogram of the samples together with the wanted pose distribution p(x) (Hint: Take care the the histogram should be scaled as a probability density function to be comparable with p(x)). How well does the histogram of samples fit with p(x) for the different choices of k? Can you imagine any problems occurring when using a uniform proposal distribution with our particular choice of p(x)?

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define the Gaussian distributions
gaussians = [(0.3, 2.0, 1.0), (0.4, 5.0, 2.0), (0.3, 9.0, 1.0)]

# Define the pose distribution p(x)
def p(x):
    return sum(w * norm.pdf(x, mu, sigma) for w, mu, sigma in gaussians)

# Define the proposal distribution q(x)
def q_uniform(x):
    return np.random.uniform(0, 15, size=int(x))

# Define the new proposal distribution q(x)
def q_gauss(x, drawSample : bool):
    if drawSample: 
        return np.random.normal(5, 4, size=int(x))
    else: 
        return norm.pdf(x, 5, 4)

# Perform the SIR algorithm
def sir(k, distibrution):
    # Compute weights
    if distibrution == 'uniform':
        # Generate initial samples
        samples = q_uniform(k)
        weights = p(samples) / q_uniform(k)
    elif distibrution == 'gauss': 
        # Generate initial samples
        samples = q_gauss(k, True)
        weights = p(samples) / q_gauss(samples, False)

    # Normalize weights
    weights /= sum(weights) 

    # Resample according to weights
    resamples = np.random.choice(samples, size=k, p=weights)

    return resamples

# Perform the SIR algorithm for different values of k
for k in [20, 100, 1000]:
    resamples = sir(k, 'uniform')

    # Calculate histogram
    counts, bins = np.histogram(resamples, bins=30)

    # Normalize counts by total number of samples
    fractions = counts / k

    # Plot histogram of fractions
    plt.bar(bins[:-1], fractions, width=np.diff(bins), alpha=0.5, label=f'k={k}')

# Plot the pose distribution
x = np.linspace(0, 15, 1000)
plt.plot(x, p(x), 'r', label='p(x)')

plt.legend()
plt.xlim(0,15)
plt.savefig('Ex2/fig1.png')
plt.show()
plt.close()



################################################
# 3.2
################################################

"""
For the next question we choose the proposal distribution q(x) to be a normal distribution with N(5,4). Again you can generate samples from such a distribution using for instance Python’s random pack- age.
Write a python program that produces a set of resampled robot poses x using the Sampling-Importance- Resampling algorithm and the above stated pose distribution p(x) and proposal distribution q(x). Show the distribution of samples after the resampling step for k = 20, 100, 1000 samples / particles. Plot a histogram of the samples together with the wanted pose distribution p(x) (Hint: Take care the the histogram should be scaled as a probability density function to be comparable with p(x)). How well does the histogram of samples fit with p(x) for the different choices of k?
"""

# Perform the SIR algorithm for different values of k
for k in [20, 100, 1000]:
    resamples = sir(k, 'gauss')
    # Calculate histogram
    counts, bins = np.histogram(resamples, bins=30)

    # Normalize counts by total number of samples
    fractions = counts / k

    # Plot histogram of fractions
    plt.bar(bins[:-1], fractions, width=np.diff(bins), alpha=0.5, label=f'k={k}')

# Plot the pose distribution
x = np.linspace(0, 15, 1000)
plt.plot(x, p(x), 'r', label='p(x)')

plt.legend()
plt.xlim(0,15)
plt.savefig('Ex2/fig2.png')
plt.show()
plt.close()
