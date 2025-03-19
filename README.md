# EZ-Diffusion-Model

1) EZ Diffusion Model Parameter Recovery
This simulate-and-recover exercise aims to assess how well the EZ diffusion model recovers key decision-making parameters across different sample sizes. Through this simulation, I was able to simulate data, apply the EZ model, and then compare the recovered parameters to the true values, in order to examine biases and squared errors. Based on this data, I was able to determine the reliability of the model's estimates 

2) Effects of Sample Size on Bias and Accuracy
The results indicate that the EZ model's parameter recovery improves as the sample size gets larger. For example, at N=10, the bias was relatively large (-0.02998), suggesting that the recovered parameters differentiate significantly from the true values. When the sample size increases to N=40, the bias decreases but remains noticeable (-0.03861), indicating a small persistent deviation from the true values. However, at N=400, the bias nearly disappears (-0.00063). This suggests that smaller sample sizes contain more deviations in their parameter estimations, whereas with larger samples, the model provides closer estimates to the true parameters.  
3) Squared Error Decreases with More Data
The squared error combines both the bias and variability, measuring the overall deviation of the estimated parameters from the true values. The results clearly demonstrate an inverse relationship between sample size and squared error. For example, for N=10, the squared error is relatively high (0.1616), but as the sample size increases to N=40, the squared error decreases (0.0921). At N=400, the squared error drops significantly (0.0011). This pattern suggests that larger sample sizes reduce the estimation error, making the EZ diffusion model more reliable in recovering true parameters. 
4) Component-wise Error Patterns
The squared errors for boundary separation (a) and non-decision time (t) provides more insight. The 'a' parameter's squared error is 0.0700 for N=10, which reduces significantly to 0.00016 for N=400, indicating that estimates for boundary separation became much more precise with a larger sample. Similarly, the 't' parameter's squared error follows the same pattern, decreasing from 0.0159(N=10) to 0.000046(N=400). This suggests that all the components of the EZ diffusion model benefit from the increased data. 
5) Personal Limitation Observed
A limitation in this experiement was computing L = np.log(R_obs/(1-R_obs)) when R_obs was equal to 0 or 1, causing log calculation issues. I decided to raise a ValueError or ZeroDivisonError for these values and then apply clipping to correct borderline values. This ensured that R_obs remianed within a safe range to avoid numerical problems. I choose this error handling method to process the data while minimizing interference.
6) Conclusion
The simulate-and-recover exercise demonstrates that the EZ diffusion model provides accurate and reliable parameter estimates only when the samples sizes are sufficiently large. Smaller samples result in high squared errors and substantial estimation variability, making results less reliable. Larger samples reduce bias and squared error, improving parameter recovery, though some bias may still persist. This suggests that moderate sample sizes might not always be sufficient. This implies that researchers using the EZ diffusion model should be caution when working with small sample sizes, as estimations may be systematically biased. Overall, the model's reliability depends on sample size, and sufficient data is essential to minimize bias and maximize accuracy. 