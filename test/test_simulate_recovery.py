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
        #Checcks that bias in estimated parameters is close to zero across many simulations
        df = self.model.simulate_recover(N=4000)

        self.assertAlmostEqual(np.mean(df['v_bias']),0,places=2,msg="Mean v_bias should be close to 0")
        self.assertAlmostEqual(np.mean(df['a_bias']),0,places=2,msg="Mean a_bias should be close to 0")
        self.assertAlmostEqual(np.mean(df['t_bias']),0,places=2,msg="Mean t_bias should be close to 0")
        
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
            self.model.inverse_equations(-0.7,0.5,0.02) #Invalid R_obs (negative)

        with self.assertRaises(ValueError):
            self.model.forward_equations(1.5,-1.0,0.2) #Invalid boundary seperation (negative)
    
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

if __name__ == "__main__":
    unittest.main()
