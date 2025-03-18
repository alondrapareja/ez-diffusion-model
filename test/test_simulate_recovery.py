import unittest
import numpy as np
import pandas as pd
from src.simulate_recovery import EZDiffusionModel 

class TestEZDiffusions(unittest.TestCase):
    
    def setUp(self):
        #Initializes a new instance of the EZDiffusionModel before each test
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



    def test_squared_errors_decrease(self):
        #Checks that the squared error decreases as N increases
        errors = []
        for N in self.model.N_values:
           df=self.model.simulate_recover(N)
           total_error=df[['v_squared_error', 'a_squared_error', 't_squared_error']].mean().sum()
           errors.append(total_error)

        self.assertTrue(errors[0]>errors[1]>errors[2],"Squared errors should decreases with larger N")

    #Tests Edge cases
    def test_divison_by_zero(self):
        #Test vision by zero
        with self.assertRaises(ZeroDivisionError):
            self.model.inverse_equations(0.0,0.5,0.02) #R_obs = 0 should raise error
    
    def test_invalid_inputs(self):
        #Tests invalid input (negative values for boundary seperation or drift rate)
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
        #Tests clamping behavior to make sure R_obs is not 0 or 1
        R_obs = np.array([0.0,1.0]) #These should get clamped
        M_obs,V_obs = 0.5,0.02
        for R in R_obs:
           R_clamped = np.clip(R,0.01,0.99)
           v_est,a_est,t_est=self.model.inverse_equations(R_clamped,M_obs,V_obs)
        self.assertGreater(v_est,0,f"Estimated drift rate should be positive for clamped R_obs={R_clamped}")
        self.assertGreater(a_est,0,f"Estimated boundary seperation should be positive for clamped R_obs={R_clamped}")
        self.assertGreater(t_est,0,f"Estimated non-decision time should be positive for clamped R_obs={R_clamped}")

    def test_large_N(self):
        N = 4000 #Large N
        df=self.model.simulate_recover(N)
        self.assertGreater(len(df),0,"The results should handle data even when N is large")
        
if __name__ == "__main__":
    unittest.main()