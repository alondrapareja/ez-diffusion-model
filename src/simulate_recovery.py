import numpy as np
import pandas as pd
import scipy.stats as stats

class EZDiffusionModel:
    def __init__(self,seed=1234):
        #Sets random seed for reproducibility
        np.random.seed(seed)  
        #Parameter ranges
        self.a_range = (0.5,2.0) # Boundary seperation #alpha
        self.v_range = (0.5,2.0) # Drift rate #v
        self.t_range = (0.1,0.5) #Nondecision time #tau
        #Sample sizes
        self.N_values = [10,40,4000]
        self.iterations = 1000   

    #Forward equations (Equations 1-3)
    def forward_equations(self,v,a,t):
        if v<0 or a <0 or t<0:
            raise ValueError("Parameters must be positive")
        y = np.exp(-a*v) 
        R_pred = 1/(y+1) #Accuracy rate / Equation 1
        M_pred = t + (a/(2*v))*((1-y)/(1+y)) #Mean RT / Equation 2
        V_pred = (a/(2*v**3))*((1-2*a*v*y-y**2)/(y+1)**2) #Variance RT / Equation 3
        return R_pred, M_pred, V_pred

    #Inverse equations to estimate parameters (Equations 4-6 )
    def inverse_equations(self,R_obs, M_obs, V_obs, epsilon=1e-5):
        #Validate inputs
        if not (0<=R_obs<=1):
            raise ValueError("R_obs must be between 0 and 1")
        if R_obs == 0.0 or R_obs == 1.0:
            raise ZeroDivisionError("R_obs can't be exactly 0 or 1 to avoid divison by 0 in log calculations")
        
        if M_obs<0 or V_obs<0:
            raise ValueError("M_obs and V_obs must be non-negative")
        
        #Clips R_obs to avoid extreme errors 
        R_obs = np.clip(R_obs,epsilon,1-epsilon)

        L = np.log(R_obs/(1-R_obs))

        #Computes v_est
        V_obs=np.clip(V_obs,epsilon,None) #Prevents division by near-zero variance
        numerator = L*(R_obs**2*L-R_obs*L+R_obs-0.5)
        v_est = np.sign(R_obs-0.5)*(numerator/V_obs)**(1/4) #Equation 4

        #Makes sure v_est is not too close to 0
        v_est=np.clip(v_est,epsilon,None)

        #Computes a_est 
        a_est = L/v_est #Equation 5
        #Computes t_est
        exp_term = np.exp(-v_est*a_est)
        t_est = M_obs-(a_est/(2*v_est))*((1-exp_term)/(1+exp_term)) #Equation 6
        return v_est, a_est, t_est

   #Simulates and Recovers Parameters
    def simulate_recover(self,N):
        results = []

        for i in range(N):
            #Generates the true parameters
            v_true = np.random.uniform(*self.v_range)
            a_true = np.random.uniform(*self.a_range)
            t_true = np.random.uniform(*self.t_range)

            #Computes predicted stats
            R_pred, M_pred, V_pred = self.forward_equations(v_true, a_true, t_true)

            #Simulates observed stats
            T_obs = min(max(1,stats.binom.rvs(N, R_pred)),N-1) #Binomial distribution. Used ChatGBT to find the binomial distribution function in Python
            R_obs = T_obs/N 
            M_obs = stats.norm.rvs(M_pred, np.sqrt(V_pred/N)) #Normal distribution. Used ChatGBT to find the normal distribution function in Python
            V_obs = stats.gamma.rvs (a=(N-1)/2, scale=2*V_pred/(N-1)) #Gamma distribution. Used ChatGBT to find the gamma function in Python 

            #Recovers estimated parameters
            v_est, a_est, t_est = self.inverse_equations(R_obs,M_obs,V_obs)

            #Computes bias 
            v_bias = v_true-v_est
            a_bias = a_true-a_est
            t_bias = t_true-t_est

            #Computes squared error
            v_squared_error = v_bias**2
            a_squared_error = a_bias**2
            t_squared_error = t_bias**2

            #Stores the results
            results.append([v_true,a_true,t_true,v_est,a_est,t_est,v_bias,a_bias,t_bias,v_squared_error,a_squared_error,t_squared_error])

        return pd.DataFrame(results, columns=["v_true","a_true","t_true","v_est","a_est","t_est","v_bias","a_bias","t_bias","v_squared_error","a_squared_error","t_squared_error"])

    #Runs simulation for different N values & saves results in CSV
    def run_simulation(self):
        for N in self.N_values:
            df = self.simulate_recover(N)
            df.to_csv(f"data/results_n{N}.csv", index=False)
            print(f"Completed simulation for N={N}")

if __name__ == "__main__":
    model = EZDiffusionModel()
    model.run_simulation()
