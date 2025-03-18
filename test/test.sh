#Runs the tests with unittest
echo "Running the test suite..."

python -m unittest discover test/ > result.log 2>&1

#Checks if the tests passed or failed and provides clear feedback 
if [ $? -eq 0 ]; then #Found this variable on Chat GBT to keep track of the exit status of the last executed command
    echo "All tests have passed successfully"
else
    echo "Some tests have failed. Check result.log for details"
fi
