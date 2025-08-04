# utils.py
import os
import pandas as pd
import torch as th
import torch.nn as nn
import geoopt
import numpy as np
import matplotlib.pyplot as plt

import warnings
from torch.optim import lr_scheduler

from focal_loss import FocalLoss  # Ensure this is implemented or installed


use_cuda = th.cuda.is_available()
device = th.device("cuda:0" if use_cuda else "cpu")

def _axat(a, x):
    r"""Returns the product :math:`A X A^\top` for each pair of matrices in a
    batch. This should give the same result as :py:`A @ X @ A.T`.
    """
    return th.einsum("...ij,...jk,...lk->...il", a, x, a)

def _atxa(a, x):
    r"""Returns the product :math:`A^\top X A` for each pair of matrices in a
    batch. 
    """
    return th.einsum("...ij,...il,...lk->...jk", a, x, a)

def _mvmt(u, w, v):
    r"""The multiplication `u @ diag(w) @ v.T`. The name stands for
    matrix/vector/matrix-transposed.
    """
    return th.einsum("...ij,...j,...kj->...ik", u, w, v)

def regularize_matrix(a, epsilon=1e-10):
    """
    Adds a small value to the diagonal elements of the matrix to regularize it.
    """
    identity = th.eye(a.size(-1), device=a.device, dtype=a.dtype) * epsilon
    return a + identity

def regularized_shrinkage(a, shrinkage=0.5, epsilon=1e-10):
    """
    Apply regularized shrinkage to the matrix.
    """
    identity = th.eye(a.size(-1), device=a.device, dtype=a.dtype) * epsilon
    return shrinkage * a + (1 - shrinkage) * identity


def calculate_class_weights(dataset):
    """
    Calculate class weights for unbalanced datasets, ensuring a tensor of shape (9).
    Missing classes will have a weight of 0, and a warning will be raised.

    Args:
        dataset: Dataset object with `y` attribute containing labels.

    Returns:
        torch.Tensor: Class weights of shape (9,).
    """
    labels = dataset.y
    total_classes = 9  # Expected number of classes
    
    # Count samples per class
    unique_classes, counts = np.unique(labels, return_counts=True)
    class_sample_count = dict(zip(unique_classes, counts))
    
    # Calculate weights
    weight = np.ones(total_classes)  # Initialize all weights to 1
    total_samples = len(labels)
    
    for cls in range(total_classes):
        if cls in class_sample_count:
            weight[cls] = 1 - (class_sample_count[cls] / total_samples)
        else:
            weight[cls] = 0
            warnings.warn(f"Class {cls} not present in the dataset. Setting weight to 0.")
    
    class_weights = th.from_numpy(weight).double()
    
    # Ensure the returned tensor has shape (9,)
    assert class_weights.shape[0] == total_classes, \
        f"Class weights tensor must have shape ({total_classes},), but got shape {class_weights.shape}"
    
    return class_weights


def plot_class_distribution(train_loader, test_loader, n_classes, folder_name="plots", covariance_mode="combo"):
    """
    Plot and display class distributions for training and test datasets.

    Args:
        train_loader: DataLoader for the training set.
        test_loader: DataLoader for the test set.
        n_classes (int): Number of classes.
        folder_name (str): Folder to save the plots and tables.
        is_combo (bool): Whether the dataset includes multiple inputs.
        covariance_mode (str): Covariance mode: "temp", "spec", or "combo".
    """
    os.makedirs(folder_name, exist_ok=True)

    # Helper function to count labels
    def count_labels(loader, n_classes):
        counts = np.zeros(n_classes, dtype=int)
        for batch in loader:
            if covariance_mode == "combo":
                _, _, labels = batch  # Dataset includes covariance matrices
            elif covariance_mode in ["temp", "spec"]:
                _, labels = batch  # Dataset excludes covariance matrices
            else:
                raise ValueError(f"Unsupported covariance_mode: {covariance_mode}")
            
            for l in labels.numpy():  # Convert to numpy for indexing
                counts[l] += 1
        return counts

    # Count labels in the train and test datasets
    train_counts = count_labels(train_loader, n_classes)
    test_counts = count_labels(test_loader, n_classes)
    class_names = train_loader.dataset.class_names

    # Bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(n_classes) - 0.2, train_counts, 0.4, label='Train', alpha=0.8, color='blue')
    plt.bar(np.arange(n_classes) + 0.2, test_counts, 0.4, label='Test', alpha=0.8, color='orange')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.yscale('log')  # Log scale for better visualization of imbalances
    plt.title('Class Distribution in Train and Test Datasets')
    plt.xticks(np.arange(n_classes), class_names, rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plot_name = os.path.join(folder_name, 'class_distribution.png')
    plt.savefig(f'{plot_name}', dpi=500)
    plt.show()

    # Create and display a table with the counts
    data = {
        'Class': class_names,
        'Train': train_counts,
        'Test': test_counts
    }
    df = pd.DataFrame(data)
    print("\nClass Distribution Table:")
    print(df)

    # Save table as an image
    fig, ax = plt.subplots(figsize=(12, 6))  # Set size for the table frame
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    plot_table_name = os.path.join(folder_name, 'class_distribution_table.png')
    plt.savefig(f'{plot_table_name}', bbox_inches='tight', dpi=500, pad_inches=0.5)
    plt.show()


def setup_loss_and_optimizer(model, 
    loss=None, gamma_loss=2.0, weights=None, 
    lr=0.001, sched="plateau", step_size=1, gamma_value=0.9, device= None
):
    """
    Set up the loss function, optimizer, and learning rate scheduler.

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        loss (str): Loss function to use. Options: "focal", "cross_entropy".
        gamma_loss (float): Gamma parameter for Focal Loss.
        weights (torch.Tensor or None): Class weights for the loss function.
        lr (float): Learning rate.
        step_lr (bool): Whether to use StepLR scheduler. If False, ExponentialLR is used.
        gamma_value (float): Gamma value for learning rate scheduler.
        device (str): Device to use, either 'cuda' or 'cpu'.

    Returns:
        loss_fn: Configured loss function.
        opti: Configured optimizer.
        scheduler: Configured learning rate scheduler.
    """
    # Set up the loss function
    if loss == "focal":
        loss_fn = FocalLoss(gamma=gamma_loss, weights=weights)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weights)

    opti = geoopt.optim.RiemannianAdam(model.parameters(), lr=lr)

    # Set up the learning rate scheduler
    if sched == "step":
        scheduler = lr_scheduler.StepLR(opti, step_size=step_size, gamma=gamma_value)
    elif sched == "exp":
        scheduler = lr_scheduler.ExponentialLR(opti, gamma=gamma_value)
    elif sched == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(opti, mode="min", factor=gamma_value, patience=step_size)
    else:
        raise ValueError(f"Unsupported scheduler: {sched}. Use 'step', 'exp', or 'plateau'.")

    print(f"Using device: {device}")
    print("Loss, optimizer, and scheduler are configured.")
    return loss_fn, opti, scheduler




