import os
import json
from datetime import datetime

import argparse
import numpy as np
import torch as th
import random

# Import from utils.py
from utils import (
    calculate_class_weights,
    setup_loss_and_optimizer
)
from dataset import get_dataloader
from train import *


# Function to save configuration as JSON
def save_config(config, folder):
    with open(os.path.join(folder, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)


# Function to read configuration from JSON
def read_config(folder):
    config_path = os.path.join(folder, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    else:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

# Function to save the results to CSV
def save_results(training_results, test_results, folder):
    # Save training results (e.g., loss, accuracy, etc.)
    training_results.to_csv(os.path.join(folder, "training_results.csv"), index=False)
    
    # Save test results (e.g., final accuracy, F1-score, etc.)
    test_results.to_csv(os.path.join(folder, "test_results.csv"), index=False)

def main(args):

    # Training options and model parameters
    nruns = args.runs
    config = {
        "mode": args.mode,
        "workers": args.workers,
        "level": args.level,
        "scaling_factor": args.scaling_factor,
        "sequence_length": args.sequence_length,
        "interpolate": args.interpolate,
        "covariance_mode": args.covariance_mode,        
        "estimator": args.estimator,        
        "embed_only": args.embed_only,
        "val_size": args.val_size,
        "n_blocks": args.n_blocks,
        "loss": args.loss,
        "gamma_loss": args.gamma_loss,
        "classifier": args.classifier,
        "batchsize": args.batchsize,
        "epochs": args.epochs,
        "lr": args.lr,
        "sched": args.sched,
        "gamma_value": args.gamma_value,
        "step_size": args.step_size
    }

    device = th.device("cuda:0" if use_cuda else "cpu")

    mode_dir = os.path.join(args.outputpath, args.covariance_mode)
    os.makedirs(mode_dir, exist_ok=True)

    for run_idx in range(nruns):  # Repeat the training for nruns

        # Fix random seed for each run for reproducibility
        seed = 13579 + run_idx
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
    

        # Load Data
        traindataloader, valdataloader, testdataloader, meta = get_dataloader(
            datapath=args.datapath,
            mode=config["mode"],
            batchsize=config["batchsize"],
            num_workers=config["workers"],
            level=config["level"],
            scaling_factor=config["scaling_factor"],
            sequence_length=config["sequence_length"],
            interpolate=config["interpolate"],
            estimator=config["estimator"],
            assume_centered=False,
            covariance_mode=config["covariance_mode"],
            val_size=config['val_size'],
            seed=1234+run_idx      
        )

        print(f"Training data samples: {len(traindataloader.sampler.indices)}")
        print(f"Validation data samples: {len(valdataloader.sampler.indices)}")
        print(f"Test data samples: {len(testdataloader.dataset)}")

        # Compute the class weights
        weights = calculate_class_weights(traindataloader.dataset)
        weights.to(device)
        print(f"Class weights: {weights}")

        run_dir = os.path.join(mode_dir, f"run{run_idx}")
        os.makedirs(run_dir, exist_ok=True)

        # Save the config file
        save_config(config, run_dir)

        # Initialize the model
        model = initialize_custom_model(
            covariance_mode=config["covariance_mode"],
            classifier=config["classifier"],
            n_blocks=config["n_blocks"],
            embed_only=config["embed_only"],            
            device=device,
            _spec=meta['ndims'],
            _temp=meta['sequencelength'],
            classes=meta['num_classes'],
        )

        # Set up loss and optimizer
        loss_fn, optimizer, scheduler = setup_loss_and_optimizer(
            model,
            loss=config["loss"],
            gamma_loss=config["gamma_loss"],
            weights=weights,
            lr=config["lr"],
            sched=config["sched"],
            step_size=config["step_size"],
            gamma_value=config["gamma_value"],
            device=device
        )

        # Training
        checkpoint_path = None
        train_loss, train_acc, val_loss, val_acc, val_fscore, best_epoch, best_model_path = train_model(
            model, traindataloader, valdataloader, 
            loss_fn, optimizer, scheduler, args.epochs, run_dir, checkpoint_path, covariance_mode=config["covariance_mode"])
        
        # Plot training curves
        plot_training_results_from_csv(run_dir, "training_results.csv")

        # Store the results
        figure_name_evaluation = os.path.join(run_dir, "evaluation_plot")
        start_epoch, min_val_loss, ckpt = load_checkpoint(model, optimizer, scheduler, best_model_path)
        loss_test, acc_test, f1_test, y_true, y_pred = evaluate_model(model, best_model_path, testdataloader, loss_fn, run_dir, covariance_mode=config["covariance_mode"], device=device)

        print(f"Run {run_idx} completed. Best model and results saved.")


if __name__ == "__main__":

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train the model with different configurations")

    # Add arguments
    parser.add_argument("--datapath", type=str, default="./data", help="Dataset path")
    parser.add_argument("--resultpath", type=str, default="./results", help="Output path for results")
    parser.add_argument("--mode", type=str, default="evaluation", choices=["unittest", "evaluation"], help="Mode of operation")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--level", type=str, default="L1C", choices=["L1C", "L2A"], help="Data level")
    parser.add_argument("--scaling_factor", type=float, default=1, help="Scaling factor")
    parser.add_argument("--sequence_length", type=int, default=45, help="Sequence length")
    parser.add_argument("--interpolate", type=int, default=0, help="Interpolation of the time series. 0: no interpolation, 1: interpolation on a fixed grid (10-day intervals)")
    parser.add_argument("--estimator", type=str, default="scm", help="Covariance estimator: 'scm' or 'oas'")
    parser.add_argument("--covariance_mode", type=str, default="spec", choices=["spec", "temp", "combo"], help="Covariance mode")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation data proportion")
    parser.add_argument("--n_blocks", type=int, default=1, help="Number of resiudal blocks")
    parser.add_argument("--embed_only", type=int, default=False, help="Use embedding only")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "cross_entropy"], help="Loss function")
    parser.add_argument("--gamma_loss", type=float, default=2, help="Gamma value for focal loss")
    parser.add_argument("--classifier", type=str, default="linear", choices=["linear", "Buseman"], help="Classifier type")
    parser.add_argument("--batchsize", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--sched", type=str, default="plateau", choices=["plateau", "stepLR"], help="Scheduler type")
    parser.add_argument("--gamma_value", type=float, default=0.2, help="Gamma value for scheduler")
    parser.add_argument("--step_size", type=int, default=15, help="Step size for scheduler")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for each configuration")

    # Parse arguments
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Create timestamp for independent directory name
    base_dir = f"Expe_{timestamp}"
    args.outputpath = os.path.join(args.resultpath, base_dir)
    
    # Run the main function
    main(args)
