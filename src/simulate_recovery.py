# 1. Select some “true” parameters (v, α, 𝜏) and a sample size N
# 2. Use Equations 1, 2, and 3 to generate “predicted” summary statistics (Rpred, Mpred, Vpred)
# 3. Use Equations 7, 8, and 9 to simulate “observed” summary statistics (Robs, Mobs, Vobs)
# 4. Use Equations 4, 5, and 6 to compute “estimated” parameters (vest, αest, 𝜏est)
# 5. Compute the estimation bias b = (v, α, 𝜏) - (vest, αest, 𝜏est) and square error b2 

# Ideally, b is close to 0 on average, and b2 should decrease when we increase N
# Certainly, b should be 0 when we set (Robs, Mobs, Vobs) = Rpred, Mpred, Vpred)

# Forward Equations/ Generate "Predicted" Summary Stats/Equations 1-3
# R_pred #predicted accurary rate
# M_pred #mean RT
# V_pred #variance RT

# Sampling Distributions/ Simulate "Observed" Summary Stats/Equations 7-9
# Generate noisy observed stats/Simulate Observed Summary Stats
# t_obs ~ Binomial (r_pred,N)
# m_obs ~ Normal (m_predicted,v_pred/N)
# v_obs ~ Gamma (((N-1)/2),((2v_pred)/N-1))

# Inverse Equations/ Compute "Estimated" Parameters/Equations 4-6
# Use inverse equations to estimate v_est, α_est, τ_est
 
# Bias and Error Analysis
# Compute bias b and squared error b^2
# b =  (v,α,τ)−(v_est, α_est, τ_est) 
# Run this simulation 1000 timkes for each sample size (N=10,40,4000), and store results 

import numpy as np
import pandas as pd
from scipy.stats import nibom, norm

#Set random seed for reproducibility
np.random.seed(42)

#Parameter ranges
a_range = (0.5,2) # Boundary seperation #alpha
v_range = (0.5,2) # Drift rate #v
t_range = (0.1,0.5) #Nondecision time #tau

#Sample sizes
N_values = [10,40,400]
iterations = 1000

#Forward equations (Equations 1-3)
def forward_equations(v,a,t):
    a  = 0.6
    v = 1.0
    t = 0.3
    y = np.exp(-a*v) #Where do it get a and v values? Does it randomly select them from paranmeter ranges above? Or do i provide a fixed value?
    R_pred = 1/(y+1) #Accuracy rate / Equation 1
    M_pred = t + (a/(2*v))*((1-y)/(1+y)) #Mean RT / Equation 2
    V_pred = (a/(2*v**3))*((1-2*a*v*y-y**2)/(y+1)**2) #Variance RT / Equation 3
    return R_pred, M_pred, V_pred

#Inverse equations to estimate parameters (Equations 4-6 )
def inverse_equations(R_obs, M_obs, V_obs): #How do you get R_obs?
    R_obs = 0.7 #Make sure 0 < R_obs < 1 to avoid division by zero? (Check Lecture on how to best handle division by 0 case)
    L = np.log(R_obs/(1-R_obs))
    #Compute v_est
    numerator = L*(R_obs**2*L-R_obs*L+R_obs-0.5)
    v_est = np.sign(R_obs-0.5)*(numerator/V_obs)**1/4 #Equation 4
    #Comoute a_est 
    a_est = L/v_est #Equation 5
    #Compute t_est
    exp_term = np.exp(-v_est*a_est)
    t_est = M_obs-(a_est/(2*v_est))*((1-exp_term)/(1+exp_term)) #Equation 6
    return v_est, a_est, t_est

#Simulate and Recover Parameters
def simulate_recover(N):
    results = []
    for _ in range(iterations):
        #Generate the true parameters
        #Compute predicted stats
        #Simulate observed stats
        #Recover estimated parameters
        #Compute bias and squared error
        v_bias = v_true-v_est
        a_bias = a_true-a_est
        t_bias = t_true-t_est