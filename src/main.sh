#Sets the directory paths
REPO_DIR="/repo/ez-diffusion-model"
SRC_DIR="$REPO_DIR/src"
DATA_DIR="$SRC_DIR/data"

#Creates the necessary directories if they don't exist
mkdir -p "$DATA_DIR"

#Sets the number of iterations
NUM_ITERATIONS=3000

#Runs the simulation-and-recovery process 3000 times
for ((i = 1; i <= NUM_ITERATIONS; i++))
do
    echo "Running iteration $i..."
    python3 "$SRC_DIR/simulate_recovery.py"
    
    #Checks if the simulation ran successfully
    if [ $? -eq 0 ]; then
        echo "Iteration $i completed successfully"
    else
        echo "Iteration $i failed. Exiting"
        exit 1
    fi
done

echo "Simulation and recovery process completed for $NUM_ITERATIONS iterations"
