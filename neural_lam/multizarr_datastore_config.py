from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union
import dataclass_wizard


@dataclass
class LatLonNames:
    """
    Class for storing latitude and longitude names.

    Attributes:
        lon (str): The name of the longitude.
        lat (str): The name of the latitude.
    """
    lon: str
    lat: str

@dataclass
class Dims:
    """
    Class for storing dimension names.

    Attributes:
        time (str): The name of the time dimension.
        level (Optional[str]): The name of the level dimension. Can be None.
        x (str): The name of the x dimension.
        y (str): The name of the y dimension.
        grid (Optional[str]): The name of the grid dimension. Can be None.
    """
    time: str
    level: Optional[str]
    x: str
    y: str
    grid: Optional[str]

@dataclass
class ZarrFile:
    """
    Class for storing Zarr related information.

    Attributes:
        path (str): The path to the Zarr file.
        dims (Dims): The dimensions of the Zarr file.
        lat_lon_names (LatLonNames): The names of the latitude and longitude.
    """
    path: str
    dims: Dims
    lat_lon_names: Optional[LatLonNames] = None

@dataclass
class State:
    """
    Class for storing state information.

    Attributes:
        zarrs (List[Zarr]): A list of Zarr files for the state.
    """
    zarrs: List[ZarrFile]
    surface_vars: Union[List[str], None]
    surface_units: Union[List[str], None]
    atmosphere_vars: Union[List[str], None]
    atmosphere_units: Union[List[str], None]
    levels: Union[List[int],None]

@dataclass
class Source:
    """
    Class for storing state or forcing collection information.

    Attributes:
        zarrs (List[Zarr]): A list of Zarr files for the forcing.
    """
    zarrs: List[ZarrFile]
    surface_vars: Union[List[str], None]
    surface_units: Union[List[str], None]
    atmosphere_vars: Union[List[str], None]
    atmosphere_units: Union[List[str], None]
    levels: Union[List[int], None]


@dataclass
class DataConfig(dataclass_wizard.YAMLWizard):
    """
    Class for storing data configuration.

    Attributes:
        name (str): The name of the configuration.
        state (State): The state information.
        surface_vars (List[str]): A list of surface variable names.
        surface_units (List[str]): A list of surface unit names.
        atmosphere_vars (List[str]): A list of atmosphere variable names.
        atmosphere_units (List[str]): A list of atmosphere unit names.
        levels (List[int]): A list of level numbers.
        forcing (Forcing): The forcing information.
    """
    name: str
    state: Source
    forcing: Source
    

if __name__ == "__main__":
    config = DataConfig.from_yaml_file("neural_lam/data_config.yaml")
    import rich
    rich.print(config)