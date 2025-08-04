import os
import time
import torch as th
from torch.optim import lr_scheduler

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE

import seaborn as sns

use_cuda = th.cuda.is_available()
device = th.device("cuda:0" if use_cuda else "cpu")
from utils import *
from model import LogEucRResNet, LogEucRResNet_Combo

def initialize_custom_model(covariance_mode, classifier, n_blocks, embed_only, device, _spec, _temp, classes):
    """
    Function to initialize the LogEucRResNet models with covariance mode supporting spec, temp, or combo.
    
    Args:
        covariance_mode (str): The mode for the covariance classifier. Options: "spec", "temp", "combo".
        classifier (str): Type of classifier to use. Options: "linear".
        n_blocks (int): Number of resiudal blocks.
        embeded_only (bool): Whether return the embeded only
        device (str): Device to use, either 'cpu' or 'cuda'.
        _spec (int): Input dimension for the spectral modality.
        _temp (int): Input dimension for the temporal modality.
        classes (int): Number of output classes.        
    """
    
    # Calculate dim1 as 80% of input dimension
    dim1_temp = int(_temp * 0.8)  # Temporal dimension
    dim1_spec = int(_spec * 0.8)  # Spectral dimension

    if covariance_mode == "combo":
        model = LogEucRResNet_Combo(
                inputdim_temp=_temp, dim1_temp=dim1_temp, 
                inputdim_spec=_spec, dim1_spec=dim1_spec,
                classes=classes, n_blocks= n_blocks, 
                embed_only=embed_only, classifier = classifier, 
            ).to(device)
        if classifier == "linear":
            print("Combo model with linear classifier")
        else:
            raise ValueError("For combo mode, only the 'linear' classifier is supported.")

    elif covariance_mode == "temp":
        model = LogEucRResNet(
                inputdim=_temp, dim1=dim1_temp, 
                classes=classes, n_blocks= n_blocks, 
                embed_only=embed_only, classifier = classifier, 
            ).to(device)
        if classifier == "linear":            
            print("Single modality - Temp model with linear classifier")
        else:
            raise ValueError("For temp mode, choose 'linear' classifier.")
    
    elif covariance_mode == "spec":
        model = LogEucRResNet(
                inputdim=_spec, dim1=dim1_spec, 
                classes=classes, n_blocks= n_blocks, 
                embed_only=embed_only, classifier = classifier, 
            ).to(device)
        if classifier == "linear":            
            print("Single modality - Spectral model with linear classifier")
        else:
            raise ValueError("For spec mode, choose 'linear' classifier.")
    
    else:
        raise ValueError("Unsupported covariance_mode. Choose from 'spec', 'temp', or 'combo'.")

    # Get number of trainable parameters
    model.eval()
    trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable total parameters: {trainable_total_params}')
    return model



def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Loads a checkpoint and resumes training from where it left off.

    Parameters:
    - model: The model to load the state dict into.
    - optimizer: The optimizer to load the state dict into.
    - scheduler: The learning rate scheduler to load the state dict into.
    - checkpoint_path: Path to the checkpoint file.

    Returns:
    - start_epoch: The epoch to resume training from.
    - best_loss: The best loss achieved before checkpointing.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at '{checkpoint_path}'")

    checkpoint = th.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer_state_dict = checkpoint['optimizer_state_dict']
    # Ensure 'step' key exists in the optimizer state
    for state in optimizer_state_dict['state'].values():
        if 'step' not in state:
            state['step'] = 0  # Initialize 'step' if missing

    optimizer.load_state_dict(optimizer_state_dict)
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint.get('loss', float('inf'))

    return start_epoch, best_loss, checkpoint

def train_one_epoch(model, train_loader, loss_fn, optimizer, device, covariance_mode):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in train_loader:

        if covariance_mode=="combo":
            x1, x2, y = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            inputs = (x1, x2)
        else:
            x, y = batch
            x = x.to(device)
            inputs = (x,)      

        y = y.to(device).long()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(*inputs)
        l = loss_fn(out, y)
        
        running_loss += l.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
        
        l.backward()
        optimizer.step()

    loss_train = running_loss / len(train_loader)    
    acc_train = correct / total
    
    return loss_train, acc_train

def validate_one_epoch(model, val_loader, loss_fn, device, covariance_mode):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true, y_pred = [], []

    for batch in val_loader:     

        if covariance_mode == 'combo':
            x1, x2, y = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            inputs = (x1, x2)
        else:
            x, y = batch
            x = x.to(device)
            inputs = (x,)
        y = y.to(device).long()
        
        with th.no_grad():
            # Forward pass
            out = model(*inputs)
            l = loss_fn(out, y)


        predicted_labels = out.argmax(1)

        running_loss += l.item()
        correct += (predicted_labels == y).sum().item()
        total += y.size(0)
              
        y_true.extend(list(y.cpu().numpy()))
        y_pred.extend(list(predicted_labels.cpu().numpy()))

    loss_val = running_loss / len(val_loader)
    acc_val = correct / total
    
    f1_val = f1_score(y_true, y_pred, average='macro')
    
    return loss_val, acc_val, f1_val, y_true, y_pred

def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs, folder_name, checkpoint_path, covariance_mode):
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, val_fscore_list = [], [], [], [], []
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    min_val_loss = float('inf')
    best_epoch = 0
    start_epoch = 0

    # Load checkpoint if exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, min_val_loss, checkpoint = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        train_loss_list = checkpoint['train_loss_history']
        train_acc_list = checkpoint['train_accuracy_history']
        val_loss_list = checkpoint['val_loss_history']
        val_acc_list = checkpoint['val_accuracy_history']
        val_fscore_list = checkpoint['val_fscore_history']
        best_epoch = checkpoint['epoch']
        print(f'Resuming training from epoch {start_epoch} with best loss {min_val_loss}')

    for epoch in range(start_epoch, epochs):
        start_time = time.time()  # Start time of the epoch

        # Training phase
        loss_train, acc_train = train_one_epoch(model, train_loader, loss_fn, optimizer, device, covariance_mode)

        train_loss_list.append(loss_train)
        train_acc_list.append(100 * acc_train)

        # Validation phase
        loss_val, acc_val, f1_val, y_true, y_pred = validate_one_epoch(model, val_loader, loss_fn, device, covariance_mode)

        val_loss_list.append(loss_val)
        val_acc_list.append(100 * acc_val)
        val_fscore_list.append(100 * f1_val)

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss_val)
        else:
            scheduler.step()
        
        print('Epoch {}/{} - Train loss: {:.4f} - Val loss: {:.4f} - Train acc: {:.2f}% - Val acc: {:.2f}% - Val F1-score: {:.2f}'.format(
            epoch + 1, epochs, loss_train, loss_val, 100 * acc_train, 100 * acc_val, 100 * f1_val))  
        
        # Save the best model
        if (epoch>int(0.1*epochs) and loss_val < min_val_loss) or epoch == int(0.1*epochs):
            min_val_loss = loss_val
            best_epoch = epoch + 1

            ckpt = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'train_loss_history': train_loss_list,
                'train_accuracy_history': train_acc_list,
                'val_loss_history': val_loss_list,
                'val_accuracy_history': val_acc_list,
                'val_fscore_history': val_fscore_list,
                'y_true': y_true,
                'y_pred': y_pred,
                'loss': min_val_loss
            }
            
            best_model_path = os.path.join(folder_name, 'best_model.pth')
            th.save(ckpt, best_model_path)
            print(f'Best model checkpoint saved at {best_model_path}')

        elapsed_time = time.time() - start_time
        print('Elapsed time: {:.2f} seconds'.format(elapsed_time))

    # Save training results to a CSV file
    results_df = pd.DataFrame({
        'epoch': np.arange(1, epochs + 1),
        'train_loss': train_loss_list,
        'train_acc': train_acc_list,
        'val_loss': val_loss_list,
        'val_acc': val_acc_list,
        'val_f1_score': val_fscore_list
    })
    results_df.to_csv(os.path.join(folder_name, 'training_results.csv'), index=False)

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list, val_fscore_list, best_epoch, best_model_path


def plot_training_results_from_csv(folder_name, csv_file):
    # Read CSV file containing the training results
    df = pd.read_csv(os.path.join(folder_name, csv_file))

    # Plot the train and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='red', linestyle='-', marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='blue', linestyle='-', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Val Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save the loss plot
    loss_plot_name = os.path.join(folder_name, 'loss_curves.png')
    plt.savefig(loss_plot_name, bbox_inches='tight', pad_inches=0.2, dpi=500)
    print(f'Loss plot saved as loss_curves.png')
    plt.close()

    # Plot the train and validation accuracy and F1-score on the second axis
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy
    fig, ax1 = plt.subplots()
    ax1.plot(df['epoch'], df['train_acc'], label='Train Accuracy', color='orange', linestyle='-', marker='o', markersize=5)
    ax1.plot(df['epoch'], df['val_acc'], label='Val Accuracy', color='green', linestyle='-', marker='x', markersize=5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Train and Val Accuracy over Epochs')
    ax1.legend(loc='upper left')
    ax1.set_ylim(50, 85)  # Set fixed y-axis range for accuracy
    
    # Create second y-axis for F1-score
    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['val_f1_score'], label='Val F1-Score', color='purple', linestyle='--', marker='s', markersize=5)
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(35, 55)  # Set fixed y-axis range for F1-score
    
    # Display grid and adjust layout
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    # Save the accuracy and F1-score plot
    perf_metrics_plot_name = os.path.join(folder_name, 'perf_metric_curves.png')
    plt.savefig(perf_metrics_plot_name, bbox_inches='tight', pad_inches=0.2, dpi=500)
    print(f'Performance metrics plot saved as perf_metric_curves.png')
    plt.close()

def evaluate_only_model(model, best_model_path, data_loader, loss_fn, covariance_mode, device):
    # Load the best model
    ckpt = th.load(best_model_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    epoch = ckpt['epoch']
    print(f'Model loaded from the best epoch {epoch}')
    print('Model loaded')
    
    loss_test, acc_test, f1_test, y_true, y_pred = validate_one_epoch(model, data_loader, loss_fn, device, covariance_mode)

    print('Test loss: {:.4f}'.format(loss_test))
    print('Test accuracy: {:.2f}%'.format(100 * acc_test))
    print('Test F1-Score: {:.2f}%'.format(100 * f1_test))

    
    return loss_test, 100 * acc_test, f1_test, y_true, y_pred

def evaluate_model(model, best_model_path, data_loader, loss_fn, folder_name, covariance_mode, device):
    # Load the best model
    ckpt = th.load(best_model_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    epoch = ckpt['epoch']
    print(f'Model loaded from the best epoch {epoch}')
    print('Model loaded')
    
    loss_test, acc_test, f1_test, y_true, y_pred = validate_one_epoch(model, data_loader, loss_fn, device, covariance_mode)

    print('Test loss: {:.4f}'.format(loss_test))
    print('Test accuracy: {:.2f}%'.format(100 * acc_test))
    print('Test F1-Score: {:.2f}%'.format(100 * f1_test))

    # Print and save classification report
    cf_report = classification_report(y_true, y_pred, digits=2, labels=np.arange(len(data_loader.dataset.class_names)), target_names=data_loader.dataset.class_names)
    print(cf_report)
    cf_report_filename = os.path.join(folder_name, f'classification_report.txt')
    with open(cf_report_filename, 'w') as f:
        f.write(cf_report)
    print(f'Classification report saved as {cf_report_filename}')

    # Compute and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred,labels=np.arange(len(data_loader.dataset.class_names)))

    # Plot and save the confusion matrix figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='', cmap='Reds', xticklabels=data_loader.dataset.class_names, yticklabels=data_loader.dataset.class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n Acc={:.2f}%'.format(100 * acc_test))
    
    plot_name = os.path.join(folder_name, 'evaluation_plot.png')
    plt.savefig(plot_name, bbox_inches='tight', pad_inches=0.2, dpi=500)
    print(f'Figure saved as {plot_name}.png')

    return loss_test, 100 * acc_test, f1_test, y_true, y_pred



