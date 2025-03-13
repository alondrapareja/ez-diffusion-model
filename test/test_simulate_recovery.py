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
    #Test sewuences of operations
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