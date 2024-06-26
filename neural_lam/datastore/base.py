import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import torch
import xarray as xr

from typing import List, Dict
import abc
import dataclasses


class BaseDatastore(abc.ABC):
    """
    Base class for weather data used in the neural-lam package. A datastore
    defines the interface for accessing weather data by providing methods to
    access the data in a processed format that can be used for training and
    evaluation of neural networks.
    
    If `is_ensemble` is True, the dataset is assumed to have an `ensemble_member` dimension.
    If `is_forecast` is True, the dataset is assumed to have a `analysis_time` dimension.
    """
    is_ensemble: bool = False
    is_forecast: bool = False

    @property
    @abc.abstractmethod
    def step_length(self) -> int:
        """
        The step length of the dataset in hours.
        
        Returns:
            int: The step length in hours.
        """
        pass

    @abc.abstractmethod
    def get_vars_units(self, category: str) -> List[str]:
        """
        Get the units of the variables in the given category.
        
        Parameters
        ----------
        category : str
            The category of the variables.

        Returns
        -------
        List[str]
            The units of the variables.
        """
        pass
    
    @abc.abstractmethod
    def get_vars_names(self, category: str) -> List[str]:
        """
        Get the names of the variables in the given category.
        
        Parameters
        ----------
        category : str
            The category of the variables.

        Returns
        -------
        List[str]
            The names of the variables.
        """
        pass
    
    @abc.abstractmethod
    def get_num_data_vars(self, category: str) -> int:
        """
        Get the number of data variables in the given category.
        
        Parameters
        ----------
        category : str
            The category of the variables.

        Returns
        -------
        int
            The number of data variables.
        """
        pass
        
    @abc.abstractmethod
    def get_dataarray(self, category: str, split: str) -> xr.DataArray:
        """
        Return the processed dataset for the given category and test/train/val-split that covers
        the entire timeline of the dataset.
        The returned dataarray is expected to at minimum have dimensions of `(time, grid_index, feature)` so
        that any spatial dimensions have been stacked into a single dimension and all variables
        and levels have been stacked into a single feature dimension.
        Any additional dimensions (for example `ensemble_member` or `analysis_time`) should be kept as separate
        dimensions in the dataarray, and `WeatherDataset` will handle the sampling of the data.
        
        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).
        split : str
            The time split to filter the dataset (train/val/test).

        Returns
        -------
        xr.DataArray
            The xarray DataArray object with processed dataset.
        """
        pass
    
    @property
    @abc.abstractmethod
    def boundary_mask(self):
        """
        Return the boundary mask for the dataset, with spatial dimensions stacked.
        Where the value is 1, the grid point is a boundary point, and where the value is 0,
        the grid point is not a boundary point.

        Returns
        -------
        xr.DataArray
            The boundary mask for the dataset, with dimensions `('grid_index',)`.
        """
        pass
    

@dataclasses.dataclass
class CartesianGridShape:
    """
    Dataclass to store the shape of a grid.
    """
    x: int
    y: int
    

class BaseCartesianDatastore(BaseDatastore):
    """
    Base class for weather data stored on a Cartesian grid. In addition
    to the methods and attributes required for weather data in general
    (see `BaseDatastore`) for Cartesian gridded source data each `grid_index`
    coordinate value is assume to have an associated `x` and `y`-value so
    that the processed data-arrays can be reshaped back into into 2D xy-gridded arrays.

    In addition the following attributes and methods are required:
    - `coords_projection` (property): Projection object for the coordinates.
    - `grid_shape_state` (property): Shape of the grid for the state variables.
    - `get_xy_extent` (method): Return the extent of the x, y coordinates for a given category of data.
    - `get_xy` (method): Return the x, y coordinates of the dataset.
    """

    @property
    @abc.abstractmethod
    def coords_projection(self) -> ccrs.Projection:
        """Return the projection object for the coordinates.

        The projection object is used to plot the coordinates on a map.

        Returns
        -------
        cartopy.crs.Projection:
            The projection object.
        """
        pass
    
    @property
    @abc.abstractmethod
    def grid_shape_state(self) -> CartesianGridShape:
        """
        The shape of the grid for the state variables.
        
        Returns
        -------
        CartesianGridShape:
            The shape of the grid for the state variables, which has `x` and `y` attributes.
        """
        pass
    
    @abc.abstractmethod
    def get_xy(self, category: str, stacked: bool) -> np.ndarray:
        """
        Return the x, y coordinates of the dataset.
        
        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).
        stacked : bool
            Whether to stack the x, y coordinates.

        Returns
        -------
        np.ndarray or tuple(np.ndarray, np.ndarray)
            The x, y coordinates of the dataset with shape `(2, N_y, N_x)` if `stacked=True` or
            a tuple of two arrays with shape `((N_y, N_x), (N_y, N_x))` if `stacked=False`.
        """
        pass
        
    def get_xy_extent(self, category: str) -> List[float]:
        """
        Return the extent of the x, y coordinates for a given category of data.
        The extent should be returned as a list of 4 floats with `[xmin, xmax, ymin, ymax]` 
        which can then be used to set the extent of a plot.
        
        Parameters
        ----------
        category : str
            The category of the dataset (state/forcing/static).
            
        Returns
        -------
        List[float]
            The extent of the x, y coordinates.
        """
        xy = self.get_xy(category, stacked=False)
        return [xy[0].min(), xy[0].max(), xy[1].min(), xy[1].max()]