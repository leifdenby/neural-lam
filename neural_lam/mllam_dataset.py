import os
import glob
import torch
import numpy as np
import datetime as dt
import random
import xarray as xr
from copy import deepcopy

from neural_lam import utils, constants


class MllamDataset(torch.utils.data.Dataset):
    def __self__(self, dataset_path, n_predicition_timesteps):
        """
        Base class for datasets used in the mllam project.
        
        To create a pytorch.Dataset for a specific model, subclass this class,
        implement the `__getitem__` method and define the `DATA_VARIABLES` dictionary
        (which should contain the names of the variables the input dataset is expected
        to contain, and the dimensions of each variable).
        """
        self.dataset_path = dataset_path
        
        self.ds = xr.open_zarr(self.dataset_path)
        self.n_input_timesteps = self.N_INPUT_TIMESTEPS
        self.n_predicition_timesteps = n_predicition_timesteps
        
        # check that the `DATA_VARIABLES` dictionary is defined and that all the
        # variables are present in the dataset
        assert hasattr(self, "DATASET_VARIABLES"), "DATASET_VARIABLES not defined"
        for var in self.DATASET_VARIABLES:
            assert var in self.ds, f"Variable {var} not found in dataset"
            for dim in self.DATASET_VARIABLES[var]:
                assert dim in self.ds[var].dims, f"Dimension {dim} not found in variable {var}"
        

    def __len__(self):
        nt = self.ds.time.shape[0]
        return nt - self.n_input_timesteps - self.n_predicition_timesteps


class GraphWeatherModelDataset(MllamDataset):
    """
    Generic pytorch.Dataset class for training datasets produced with mllam-data-prep

    For graph based models there three kinds of data used:

    |---------|---------------|------------------------|----------------------------|
    | type    | time-varying  | predicted by the model | used as input to the model |
    |---------|---------------|------------------------|----------------------------|
    | state   | yes           | yes                    | yes                        |
    | static  | no            | no                     | yes                        |
    | forcing | yes           | no                     | yes                        |
    |---------|---------------|------------------------|----------------------------|

    Each of these kinds of data are expected to be represented by separate variables
    in the dataset. Each kind of data may contain a number of features (i.e. each variable
    has a "feature" coordinate, given by the name `{data_kind}_feature`). The spatial coordinates are
    collapsed to a single `grid_index` dimension. In addition the time-varying input variables
    have a `time` coordinate. Finally, the dataset is expected to be stored in a zarr format.

    All of these properties are given by the `MODEL_INPUTS` class variable.

    To construct the input-output pairs for the model, the dataset is sampled along the time
    dimensions to create the following input-output pairs:
        inputs (X):
        - init_states: (2, num_grid_nodes, d_features)
            the initial state (this model uses 2 time steps)

        - forcing_features: (pred_steps, num_grid_nodes, d_forcing),
            the forcing features (again the same as the AR-number)

        - static_features: (num_grid_nodes, d_static_f)
        
        outputs (Y):
        - target_states: (pred_steps, num_grid_nodes, d_features)
            the target state (the number of steps being the AR-number)
            
    So that the model f attempts to predict the target states given the initial states, forcing
    and static features, e.g. f(X) = Y_hat, and the loss is computed as the difference between
    Y_hat and Y.

    The initial states are sampled at times t-2 and t-1, and the target states are sampled
    at times t0, t1, t2, t3 (the number of steps being the AR-number, or number of auto-
    regressive steps).

    init_state    s(t-2)   s(t-1)
    target_state                    s(t0)   s(t1)   s(t2)   s(t3)

    The forcing features are windowed over the time dimension by stacking along the feature
    dimension, so that the forcing features for the first time step are the forcing features for
    the time steps t-2, t-1, t0. The forcing features for the second time
    step are the forcing features for the time steps t-1, t0, t1, and so on.

    forcing                         f(t-2)  f(t-1)  f(t0)   f(t1)
                                    f(t-1)  f(t0)   f(t1)   f(t2)
                                    f(t0)   f(t1)   f(t2)   f(t3)
                                    

    TODO:
    - implement standardization
    - implement random subsampling
    - implement training/val/test split
    """
    DATASET_VARIABLES = dict(
        state=["time", "grid_index", "state_feature"],
        static=["grid_index", "static_feature"],
        forcing= ["time", "grid_index", "forcing_feature"]
    )
    
    N_INPUT_TIMESTEPS = 2
        
        
    def __getitem__(self, idx):
        ds_sample = self.ds.isel(time=slice(idx, idx + self.n_input_timesteps + self.n_predicition_timesteps))
        
        da_init_states = ds_sample.isel(time=slice(0, self.n_input_timesteps)).state
        da_target_states = ds_sample.isel(time=slice(self.n_input_timesteps, self.n_input_timesteps + self.n_predicition_timesteps)).state
        
        # each prediction will always be made with n_input_timesteps to predict the
        # next timestep, so we need n_input_timesteps + 1 forcing features aligned
        # with the target
        das_forcing = []
        for i in range(self.n_predicition_timesteps + 1):
            das_forcing.append(ds_sample.isel(time=slice(i, i + self.n_input_timesteps)).forcing)
        da_forcing = xr.concat(das_forcing, dim='forcing_features')
        
        da_static_features = ds_sample.static
        
        # convert each data array to a tensor
        init_states = torch.tensor(da_init_states.values)
        target_states = torch.tensor(da_target_states.values)
        forcing_windowed = torch.tensor(da_forcing.values)
        static_features = torch.tensor(da_static_features.values)
        
        return init_states, target_states, static_features, forcing_windowed