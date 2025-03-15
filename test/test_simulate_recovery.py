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

if __name__ == "__main__":
    unittest.main()
