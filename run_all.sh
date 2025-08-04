#!/bin/bash



# Define paths and configurations
DATAPATH="../data"           # Path to the data (path where the data will be stored when running the script for the first time)
RESULTPATH="../results"      # Path to the results 
MODE="evaluation"           # Mode of the script (evaluation or unittest). Use evaluation to reproduce the results in the paper

WORKERS=4                   # Number of workers for DataLoader
LEVEL="L1C"                 # Level of the data (tested only on L1C)   
SCALING_FACTOR=0.0001       # Scaling factor for the data (divided by 10000 to get the original reflectance)
INTERPOLATE=1               # Interpolation flag
SEQUENCE_LENGTH_LIST=(45)   # Length of the sequence (if no interpolation)
ESTIMATOR_LIST=("oas")      # Estimation for the covariance matrix: "oas" or "scm"
VAL_SIZE=0.1                # Validation size

N_BLOCKS_LIST=(1)           # List of residual blocks to iterate over
EPOCHS=100                  # Number of epochs
BATCH_SIZE=128              # Batch size
LOSS="focal"                # Loss function
GAMMA_LOSS=2                # Gamma value for the focal loss
LR_LIST=(1e-2)              # Learning rates to iterate over
SCHEDULER="plateau"         # Scheduler type: "plateau" for ReduceLROnPlateau, "step" for StepLR, "exp" for ExponentialLR
GAMMA_VALUE=0.2             # Gamma value for all the schedulers
STEP_SIZE=10                # Step size for StepLR and for the patience of ReduceLROnPlateau

RUNS=5                      # Number of runs per configuration

# Covariance modes to iterate over
COVARIANCE_MODES=("spec" "temp" "combo")


# Loop over the number of residual blocks
for SEQUENCE_LENGTH in "${SEQUENCE_LENGTH_LIST[@]}"; do
    # Loop over estimators
    for ESTIMATOR in "${ESTIMATOR_LIST[@]}"; do
        # Loop over learning rates
        for LR in "${LR_LIST[@]}"; do
            # Loop over the number of residual blocks
            for N_BLOCKS in "${N_BLOCKS_LIST[@]}"; do
                # Loop over covariance modes
                for COV_MODE in "${COVARIANCE_MODES[@]}"; do

                    # Call the Python script with arguments
                    python run_script.py \
                        --datapath ${DATAPATH} \
                        --resultpath ${RESULTPATH} \
                        --mode ${MODE} \
                        --workers ${WORKERS} \
                        --level ${LEVEL} \
                        --scaling_factor ${SCALING_FACTOR} \
                        --sequence_length ${SEQUENCE_LENGTH} \
                        --interpolate ${INTERPOLATE} \
                        --estimator ${ESTIMATOR} \
                        --covariance_mode ${COV_MODE} \
                        --val_size ${VAL_SIZE} \
                        --n_blocks ${N_BLOCKS} \
                        --embed_only 0 \
                        --loss ${LOSS} \
                        --gamma_loss ${GAMMA_LOSS} \
                        --classifier "linear" \
                        --batchsize ${BATCH_SIZE} \
                        --epochs ${EPOCHS} \
                        --lr ${LR} \
                        --sched ${SCHEDULER} \
                        --gamma_value ${GAMMA_VALUE} \
                        --step_size ${STEP_SIZE} \
                        --runs ${RUNS}   >> output_python.log 2>&1 
                
                
                    echo "Experiment with N_BLOCKS=${N_BLOCKS}, COVARIANCE_MODE=${COV_MODE}, SEQUENCE_LENGTH=${SEQUENCE_LENGTH}, LR=${LR}, ESTIMATOR=${ESTIMATOR} complete!"
                done
            done
        done
    done
done

echo "All experiments completed!"
