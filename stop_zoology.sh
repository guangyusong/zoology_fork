# Find the PID of the process
PID=$(ps aux | grep '[p]ython -m zoology.launch zoology/experiments/arxiv24_based_figure2/configs.py' | awk '{print $2}')

# Check if PID is not empty and kill the process
if [ -z "$PID" ]; then
    echo "Process not found."
else
    echo "Killing process with PID: $PID"
    kill $PID
fi
