import breizhcrops
import os
import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy import interpolate
from interp1d import interp1d
from pyriemann.utils.covariance import covariances
from breizhcrops.datasets.breizhcrops import BANDS
from sklearn.model_selection import train_test_split

# Import from utils.py
from utils import (
    regularized_shrinkage,
)


class TS_COV(Dataset):
    def __init__(self, datapath, cache_dir, breizhcrops_dataset, transformer, estimator='scm', assume_centered=False, mode="spec"):
        """
        Initialize the TS_COV dataset class using BreizhCrops dataset.

        Args:
            datapath (str): Path to the BreizhCrops dataset directory.
            breizhcrops_dataset (Dataset): An instance of the BreizhCrops dataset.
            transformer (callable): Transformation function for time-series data.
            estimator (str): Covariance estimator type (default: 'scm').
            assume_centered (bool): Assumed data centered (default: False).
            mode (str): Covariance mode: "temp", "spec", or "combo" (default: "spec").
        """
        self.breizhcrops_dataset = breizhcrops_dataset
        self.datapath = datapath
        self.transformer = transformer
        self.estimator = estimator
        self.assume_centered = assume_centered
        self.mode = mode
        self.cache_dir = cache_dir
        self.interpolate = self.transformer.interpolate

        self.x, self.y, self.id, self.class_names = self.load_data_from_breizhcrops()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        Returns the covariance matrices and label for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Covariance matrices (x1 and/or x2) and label (y).
        """

        y_sample = self.y[idx]
        y = th.from_numpy(np.array(y_sample)).long()

        # Load the time series data and apply transformations
        time_series = self.x[idx]        
        time_series_id = self.id[idx]
        
        # Compute covariances based on the mode and transformed time series
        if self.mode == "combo" or self.mode == "temp":
            # Temporal covariance calculation
            #cov_temp = self.calculate_covariance(time_series, mode="temp")
            cov_temp = self.get_covariance_matrices(time_series, time_series_id, mode="temp")

        if self.mode == "combo" or self.mode == "spec":
            # Spectral covariance calculation
            #cov_spec = self.calculate_covariance(time_series, mode="spec")
            cov_spec = self.get_covariance_matrices(time_series, time_series_id, mode="spec")

        if self.mode == "combo":
            return cov_temp, cov_spec, y
        elif self.mode == "temp":
            return cov_temp, y
        elif self.mode == "spec":
            return cov_spec, y
        else:
            raise NameError(f"Mode {self.mode} does not exist. Options include 'temp', 'spec', and 'combo'.")

    def load_data_from_breizhcrops(self):
        """
        Loads and transforms data from the BreizhCrops dataset instance.
        """
        series = []
        labels = []
        ids = []

        for idx in range(len(self.breizhcrops_dataset)):
            sample = self.breizhcrops_dataset[idx]
            time_series, label, id = sample[0], sample[1], sample[2]

            series.append(time_series.numpy())  # Convert to NumPy for covariance calculation
            labels.append(label)
            ids.append(id)    

        class_mapping_file = os.path.join(self.datapath, "classmapping.csv")

        if os.path.exists(class_mapping_file):
            class_mapping = pd.read_csv(class_mapping_file)
            class_names = class_mapping['classname'].unique().tolist()
        else:
            raise FileNotFoundError(f"Class mapping file not found at {class_mapping_file}")

        return series, labels, ids, class_names
    
    def get_covariance_matrices(self, time_series, idx, mode):
        """
        Load or compute the covariance matrix based on the mode (either 'temp' or 'spec').

        Args:
            time_series (np.ndarray): The transformed time series data.
            mode (str): Covariance mode, either 'temp' or 'spec'.

        Returns:
            th.Tensor: Covariance matrix.
        """
        cache_file = os.path.join(self.cache_dir, f"{mode}_{idx}_cov_matrix.pt")
        
        if self.interpolate and os.path.exists(cache_file):
            # Load the saved covariance matrix
            cov_matrix = th.load(cache_file, weights_only=True)            
        else:
            # Compute the covariance matrix
            time_series = self.transformer.transform(time_series)
            time_series = np.asarray(time_series.unsqueeze(0))
            cov_matrix = self.calculate_covariance(time_series, mode)

            if self.interpolate:
                # Save the covariance matrix for future use
                th.save(cov_matrix, cache_file)
                
        return cov_matrix

    def calculate_covariance(self, time_series, mode):
        """
        Calculate covariance matrices (either spectral or temporal).

        Args:
            time_series (np.ndarray): The transformed time series data.
            mode (str): Covariance mode, either 'temp' or 'spec'.

        Returns:
            th.Tensor: Covariance matrix.
        """
        if mode == "temp":
            # Temporal covariance calculation (n-temp x n-temp)
            cov = covariances(time_series, estimator=self.estimator, assume_centered=self.assume_centered)
        elif mode == "spec":
            # Spectral covariance calculation (n-spec x n-spec)
            time_series_transposed = np.transpose(time_series, (0, 2, 1))  # Shape: (spec, temp)
            cov = covariances(time_series_transposed, estimator=self.estimator, assume_centered=self.assume_centered)
        else:
            raise ValueError(f"Invalid mode '{mode}', choose from 'temp' or 'spec'.")
        
        cov = th.from_numpy(cov)
        if self.estimator != "oas":
            cov = regularized_shrinkage(cov,shrinkage=0.99,epsilon=1e-5)
        return cov.to(th.double)


class TS_COV_Transform:
    def __init__(self, level, selected_bands=None, scaling_factor=1e-4, sequencelength=45, interpolate=False):
        """
        Initializes the transformer with the necessary parameters.

        Args:
            level (str): The data level (e.g., "L1C" or "L2A").
            selected_bands (list, optional): A list of band names to select for transformation. If None, default bands will be used.
            scaling_factor (float): Scaling factor for reflectance values (default: 1e-4).
            sequencelength (int): The desired sequence length for time-series data (default: 45).
            interpolate (bool): Whether to interpolate on fixed grid-size (default: False).
        """
        self.level = level
        self.selected_bands = selected_bands
        self.scaling_factor = scaling_factor
        self.interpolate = interpolate

        self.target_doy = np.arange(15,351,10) # every 10 days from mid-January to mid-December

        if self.interpolate:
            self.sequencelength = self.target_doy.shape[0]  # Fixed sequence length for interpolation
        else:
            self.sequencelength = sequencelength

        # Define available bands per level
        self.bands = BANDS.get(level, [])
        if not self.bands:
            raise ValueError(f"Invalid level: {level}. Please specify a valid level with available bands.")

        # Default band selection if none provided
        if self.selected_bands is None:
            if level == "L1C":
                #-- self.selected_bands = ['B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A']
                self.selected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'] 
            elif level == "L2A":
                self.selected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
            else:
                raise ValueError(f"Unsupported level: {level}. Please provide a valid level or band selection.")
        
        # Get indices of the selected bands
        self.selected_band_idxs = [self.bands.index(b) for b in self.selected_bands]
    
    def transform(self, x):
        """
        Transforms the input time series data.

        Args:
            x (th.Tensor or np.ndarray): Input time-series data (samples, time_steps, bands).

        Returns:
            th.Tensor: Transformed time-series data, reshaped and normalized.
        """
        if isinstance(x, th.Tensor):
            x = x.numpy()

        # Scale reflectance values to [0, 1] and select the relevant bands
        doa_x = [pd.to_datetime(ddate,unit='ns') for ddate in x[:, self.bands.index('doa')]]
        reference_date = datetime(doa_x[0].year,1,1)
        doy =[(ddate-reference_date).days+1 for ddate in doa_x]

        x = x[:, self.selected_band_idxs] * self.scaling_factor

        if self.interpolate:
            # Interpolate the time series data on a fixed grid size defined by self.target_doy (10-days intervals)
            x = th.from_numpy(x).type(th.FloatTensor)
            doy = th.from_numpy(np.array(doy)).type(th.FloatTensor)
            doy_new = th.from_numpy(self.target_doy).type(th.FloatTensor)
            doy_new = doy_new.unsqueeze(0).expand(x.shape[-1],-1)
            x = interp1d(doy, x.T, doy_new)
            x = x.T    
        else: # No interpolation
            # Ensure that the sequence length is met, with or without replacement
            replace = False if x.shape[0] >= self.sequencelength else True
            idxs = np.random.choice(x.shape[0], self.sequencelength, replace=replace)
            idxs.sort()

            # Extract the desired sequence
            x = x[idxs]

            # Torch tensor
            x = th.from_numpy(x).type(th.FloatTensor)

        return x 



def get_dataloader(datapath, mode, batchsize, preload_ram=True, num_workers=0,
                    level="L1C", scaling_factor=1e-2, sequence_length=45,
                    estimator='scm', assume_centered = False, covariance_mode="combo",
                    interpolate=False,
                    val_size=0.1, seed = 1234
                   ):
    """
    Set up DataLoaders using the TS_COV dataset, which computes covariances for BreizhCrops data.

    Args:
        datapath (str): Path to the dataset directory.
        mode (str): Mode for the dataset split ("evaluation1", etc.).
        batchsize (int): Batch size for DataLoader.
        preload_ram (bool): Whether to preload data into RAM.
        level (str): Data level (e.g., "L1C" or "L2A").
        estimator (str): Covariance estimator type.
        assume_centered (bool): Assumed data centered (default: False)
        covariance_mode (str): Covariance mode: "temp", "spec", or "combo".
        interpolate (bool): Whether to interpolate on fixed grid-size (default: False).
        val_size (float): Validation set size (default: 0.1).

    Returns:
        traindataloader, testdataloader, meta: DataLoaders for train and test datasets, and metadata.
    """
    print(f"Setting up datasets in {os.path.abspath(datapath)}, level {level}")
    datapath = os.path.abspath(datapath)
    cache_dir = os.path.join(datapath, "covariance_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    def transform(x):
        return th.from_numpy(x).type(th.FloatTensor)

    if mode == "unittest":
        belle_ile = breizhcrops.BreizhCrops(region="belle-ile", root=datapath, transform=transform, preload_ram=preload_ram, level=level)
    elif mode == "visu":
        frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath, transform=transform, preload_ram=preload_ram, level=level)
    else:
        frh01 = breizhcrops.BreizhCrops(region="frh01", root=datapath, transform=transform, preload_ram=preload_ram, level=level)
        frh02 = breizhcrops.BreizhCrops(region="frh02", root=datapath, transform=transform, preload_ram=preload_ram, level=level)
        frh03 = breizhcrops.BreizhCrops(region="frh03", root=datapath, transform=transform, preload_ram=preload_ram, level=level)

        # Assert that all regions have the same number of classes
        assert len(frh01.classes) == len(frh02.classes), "Different number of classes in frh01 and frh02"
        assert len(frh02.classes) == len(frh03.classes), "Different number of classes in frh02 and frh03"

        frh04 = breizhcrops.BreizhCrops(region="frh04", root=datapath, transform=transform, preload_ram=preload_ram, level=level)

    # Split datasets based on mode
    if mode == "evaluation":
        train_breizhcrops = th.utils.data.ConcatDataset([frh01, frh02, frh03])
        test_breizhcrops = frh04
    elif mode == "unittest":
        train_breizhcrops = belle_ile
        test_breizhcrops = belle_ile
    elif mode == "visu":
        train_breizhcrops = frh02
        test_breizhcrops = frh02
    else:
        raise ValueError("only --mode 'unittest' or 'evaluation' allowed")

    transformer = TS_COV_Transform(level=level, scaling_factor=scaling_factor, sequencelength=sequence_length, interpolate=interpolate)

    # Wrap the BreizhCrops datasets with the TS_COV dataset to compute covariances
    train_dataset = TS_COV(datapath=datapath, cache_dir=cache_dir, breizhcrops_dataset=train_breizhcrops, transformer=transformer, estimator=estimator, assume_centered = assume_centered, mode=covariance_mode)
    test_dataset = TS_COV(datapath=datapath, cache_dir=cache_dir, breizhcrops_dataset=test_breizhcrops, transformer=transformer, estimator=estimator, assume_centered = assume_centered, mode=covariance_mode)

    # Subsampler
    indices = list(range(0,len(train_dataset)))    
    y_train = train_dataset.y
    y_train = [tensor.item() for tensor in y_train]     
    train_indices, val_indices = train_test_split(indices,test_size=val_size,random_state=seed,stratify=y_train,shuffle=True)
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # DataLoaders for the covariance-transformed datasets
    traindataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batchsize, num_workers=num_workers, drop_last=True)
    valdataloader = DataLoader(train_dataset, sampler=val_sampler, batch_size=batchsize, num_workers=num_workers)
    testdataloader = DataLoader(test_dataset, batch_size=batchsize, num_workers=num_workers, shuffle=False)

    if mode == "unittest":
        num_classes = len(belle_ile.classname)
        class_names = belle_ile.classname
    elif mode == "visu":
        num_classes = len(frh02.classname)
        class_names = frh02.classname
    else:
        num_classes = len(frh01.classname)
        class_names = frh01.classname

    # Metadata: 10 dimensions for the 10 bands used in the transformation
    meta = dict(
        ndims=10,  # Updated to 10 dimensions/bands
        num_classes=num_classes,
        sequencelength=transformer.sequencelength,
        proto = int(num_classes+1),
        class_names=class_names
    )

    return traindataloader, valdataloader, testdataloader, meta

    

