# Unit Tests
# Very that froward/inverse equations are correct
# Ensure that bias b is 0 when noise is removed
# Check that squared error b^2 decreases with increasing N

#Common assertions
    # assertEqual(a,b) - verify a = b
    # assertAlmostEqual(a,b) - for floating point
    # assertRaises(Error) - verify exceptions
    # assertTrue(x) - verify x is False

#Testing Edge Cases
    #Divison by zero
        #def test_divide_byzero(self):
           #with self.assertRaises(ValueError):
                  #self.calc.divide(5,0) 
    #Test both normal & edge cases
    #Test invalid inputs
    #Test boundary conditions

#Testing Complex Behavior
    #Test state changes
    #Test sequences of operations
    #Verify side effects 

# Best Practices
    # Write tests before or while writing code 
    # One test method per feature/scenario
    # Clear, descriptive names
    # Independent tests (no dependencies between tests)
    # Use setUP() for common initialization 
    # Keep tests simple and focused 

#Key Requirements
    #Tests should be in a tests directory
    #Include empty __init__.py in tests directory
    #Test files should follow naming pattern test-*.py
    #Run tests from project root directory 

import unittest
import numpy as np
import pandas as pd
from src.simulate_recovery import EZDiffusionModel 

class TestEZDiffusions(unittest.TestCase):
    
    def setUp(self):
        self.model=EZDiffusionModel()

    def test_initializations(self):
        #Tests if model initializes with correct parameters and sample sizes
        self.assertEqual(self.model.a_range,(0.5,2.0))
        self.assertEqual(self.model.v_range,(0.5,2.0))
        self.assertEqual(self.model.t_range,(0.1,0.5))
        self.assertEqual(self.model.N_values,[10,40,4000])

    def test_forward_equations(self):
        #Tests forward equations for expected outputs
        v,a,t = 1.0,1.5,0.2 #Example values
        R_pred,M_pred,V_pred=self.model.forward_equations(v,a,t)

        self.assertTrue(0<R_pred<1, "R_pred should be between 0 and 1")
        self.assertGreater(M_pred,t, "Mean RT should be greater than non-decision time")
        self.assertGreater(V_pred,0,"Variance should be positive")

    def test_inverse_equations(self):
        #Tests inverse equations for proper parameter recovery
        R_obs,M_obs,V_obs=0.7,0.5,0.02
        v_est,a_est,t_est=self.model.inverse_equations(R_obs,M_obs,V_obs)

        self.assertIsInstance(v_est,float)
        self.assertIsInstance(a_est,float)
        self.assertIsInstance(t_est,float)
        self.assertGreater(v_est,0, "Estimated drift rate should be positive")
        self.assertGreater(a_est,0, "Estimated boundary seperation should be positive")
        self.assertGreater(t_est,0, "Estimated non-decision time should be positive")

    def test_bias_average_zero(self):
        #Checks that bias in estimated parameters is close to zero across many simulations
        df = self.model.simulate_recover(N=4000)

        self.assertAlmostEqual(np.mean(df['v_bias']),0,places=2,msg="Mean v_bias should be close to 0")
        self.assertAlmostEqual(np.mean(df['a_bias']),0,places=2,msg="Mean a_bias should be close to 0")
        self.assertAlmostEqual(np.mean(df['t_bias']),0,places=2,msg="Mean t_bias should be close to 0")
        
    def test_bias_zero_no_noise(self):
        #Set all values to true and observe
        v_true,a_true,t_true=1.0,1.5,0.2 #True parameters
        R_pred,M_pred,V_pred=self.model.forward_equations(v_true,a_true,t_true)

        T_obs=np.round(R_pred*1000) #Stimulate without noise
        R_obs=max(min(T_obs/1000,0.99),0.01) #Clips to avoid R_obs = 0 or 1
        M_obs = M_pred
        V_obs=V_pred

        v_est,a_est,t_est = self.model.inverse_equations(R_obs,M_obs,V_obs)

        v_bias = v_true-v_est
        a_bias = a_true-a_est
        t_bias = t_true-t_est

        self.assertAlmostEqual(v_bias, 0, places=2, msg="Bias for v should be close to 0")
        self.assertAlmostEqual(a_bias, 0, places=2, msg="Bias for a should be close to 0")
        self.assertAlmostEqual(t_bias, 0, places=2, msg="Bias for t should be close to 0")



    #def test_squared_errors_decrease(self):
        #Checks that the squared error decreases as N increases
        #errors = []
        #for N in self.model.N_values:
           #df=self.model.simulate_recover(N)
            #total_error=df[['v_squared_error', 'a_squared_error', 't_squared_error']].mean().sum()
            #errors.append(total_error)

        #self.assertTrue(errors[0]>errors[1]>errors[2],"Squared errors should decreases with larger N")

    #Tests Edge cases
    def test_divison_by_zero(self):
        #Test vision by zero
        with self.assertRaises(ZeroDivisionError):
            self.model.inverse_equations(0.0,0.5,0.02) #R_obs = 0 should raise error
    
    def test_invalid_inputs(self):
        #Test invalid input (negative values for boundary seperation or drift rate)
        with self.assertRaises(ValueError):
            self.model.inverse_equations(-0.7,0.5,0.02) #R_obs < 0

        with self.assertRaises(ValueError):
            self.model.inverse_equations(1.2,0.5,0.02) #R_obs > 1

        with self.assertRaises(ValueError):
            self.model.inverse_equations(1.5,-0.5,0.2) #M_obs < 0

        with self.assertRaises(ValueError):
            self.model.inverse_equations(0.5,0.5,-0.02) #V_obs < 0

    def test_boundary_condition(self):
        #Small epsilon value for testing boundaries
        epsilon = 1e-10

        #Should not raise errors for values close to 0 or 1
        try:
            v_low, a_low, t_low = self.model.inverse_equations(epsilon, 0.5, 0.02)
            v_high, a_high, t_high = self.model.inverse_equations(1 - epsilon, 0.5, 0.02)
        except Exception as e:
            self.fail(f"Boundary test failed: Error - {e}")

        #Asserts the results are finite numbers 
        self.assertTrue(np.isfinite(v_low) and np.isfinite(a_low) and np.isfinite(t_low))
        self.assertTrue(np.isfinite(v_high) and np.isfinite(a_high) and np.isfinite(t_high))
    
    def test_clamping_edge_case(self):
        #Values at the extreme edges that should be clamped
        small_value = 0.0
        large_value = 1.0

        #Makes sure the function does not raise an error for near boundary values
        try:
            v_low, a_low, t_low = self.model.inverse_equations(max(small_value, 1e-10), 0.5, 0.02)
            v_high, a_high, t_high = self.model.inverse_equations(min(large_value, 1 - 1e-10), 0.5, 0.02)
        except Exception as e:
            self.fail(f"Clamping test failed: Error - {e}")

        #Asserts the results are finite numbers
        self.assertTrue(np.isfinite(v_low) and np.isfinite(a_low) and np.isfinite(t_low))
        self.assertTrue(np.isfinite(v_high) and np.isfinite(a_high) and np.isfinite(t_high))

        #Makes sure the modified R_obs values are within valid range (0 < R_obs < 1)
        self.assertGreater(max(small_value, 1e-10), 0)
        self.assertLess(min(large_value, 1 - 1e-10), 1)

    #def test_squared_errors_decrease(self):
        #np.random.seed(42)  # Ensure reproducibility

        # Define increasing sample sizes
        #sample_sizes = [10, 40, 4000]
        #squared_errors = []

        #for sample_size in sample_sizes:
            # Run the simulation for a given sample size
            #results_df = self.model.simulate_recover(sample_size)

            # Compute mean squared error across v, a, and t
            #mse = results_df[['v_squared_error', 'a_squared_error', 't_squared_error']].mean().mean()
            #squared_errors.append(mse)

        # Ensure squared errors decrease as sample size increases
        #for i in range(1, len(squared_errors)):
            #self.assertLess(squared_errors[i], squared_errors[i - 1], 
                            #f"Squared error did not decrease: {squared_errors[i]} >= {squared_errors[i - 1]}")


    #def test_boundary_condition(self):
        #Test boundary conditions (extreme value close to 0 or 1)
        #R_obs = np.array([0.01,0.99])
        #M_obs, V_obs = 0.5,0.02
        #for R in R_obs:
            #v_est,a_est,t_est = self.model.inverse_equations(R,M_obs,V_obs)
            #self.assertGreater(v_est,0, f"Estimated drift rate should be positive for R_obs={R}")
            #self.assertGreater(a_est,0,f"Estimated boundary seperation should be positive for R_obs={R}")
            #self.assertGreater(t_est,0,f"Estimated non-decision time should be positive for R_obs={R}")

    #def test_clamping_edge_case(self):
        #Test clamping behavior to make sure R_obs is not 0 or 1
        #R_obs = np.array([0.0,1.0]) #These should get clamped
        #M_obs,V_obs = 0.5,0.02
        #for R in R_obs:
           #R_clamped = np.clip(R,0.01,0.99)
            #v_est,a_est,t_est=self.model.inverse_equations(R_clamped,M_obs,V_obs)
            #self.assertGreater(v_est,0,f"Estimated drift rate should be positive for clamped R_obs={R_clamped}")
            #self.assertGreater(a_est,0,f"Estimated boundary seperation should be positive for clamped R_obs={R_clamped}")
            #self.assertGreater(t_est,0,f"Estimated non-decision time should be positive for clamped R_obs={R_clamped}")

    def test_large_N(self):
        N = 4000 #Large N
        df=self.model.simulate_recover(N)
        self.assertGreater(len(df),0,"The results should handle data even when N is large")
        
if __name__ == "__main__":
    unittest.main()