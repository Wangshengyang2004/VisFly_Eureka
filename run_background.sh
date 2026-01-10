#!/bin/bash
# Run batched experiments via CLI overrides
# Each experiment runs in the background with its own log file

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to run an experiment
run_experiment() {
    local exp_name=$1
    local hydra_overrides=$2
    local log_file="logs/${exp_name}.log"
    
    echo "=========================================="
    echo "Starting experiment: ${exp_name}"
    echo "Overrides: ${hydra_overrides}"
    echo "Log file: ${log_file}"
    echo "=========================================="
    
    nohup python main.py ${hydra_overrides} > "${log_file}" 2>&1 &
    local pid=$!
    echo "Experiment '${exp_name}' started with PID: ${pid}"
    echo "Monitor with: tail -f ${log_file}"
    echo ""
    
    # Store PID for later reference
    echo "${pid}" > "logs/${exp_name}.pid"
}

# Experiment 1: Original config (hover with eureka mode - defaults)
run_experiment "exp1_hover_eureka" ""

# Experiment 2: Flip env with eureka mode
run_experiment "exp2_flip_eureka" "envs=flip"

# Experiment 3: Navigation env (from envs/, not VisFly) with eureka mode
run_experiment "exp3_navigation_eureka" "envs=navigation"

# Experiment 4: Hover with temptuner mode
run_experiment "exp4_hover_temptuner" "mode=temptuner"

# Experiment 5: Flip with temptuner mode
run_experiment "exp5_flip_temptuner" "envs=flip mode=temptuner"

# Experiment 6: Navigation with temptuner mode
run_experiment "exp6_navigation_temptuner" "envs=navigation mode=temptuner"

echo "=========================================="
echo "All 6 experiments have been started!"
echo "=========================================="
echo ""
echo "To check running processes:"
echo "  ps aux | grep 'python main.py'"
echo ""
echo "To view logs:"
echo "  tail -f logs/exp1_hover_eureka.log"
echo "  tail -f logs/exp2_flip_eureka.log"
echo "  tail -f logs/exp3_navigation_eureka.log"
echo "  tail -f logs/exp4_hover_temptuner.log"
echo "  tail -f logs/exp5_flip_temptuner.log"
echo "  tail -f logs/exp6_navigation_temptuner.log"
echo ""
echo "To check PIDs:"
echo "  cat logs/*.pid"
echo ""
echo "To kill all experiments:"
echo "  pkill -f 'python main.py'"
