# Standard library
from pathlib import Path

# First-party
from neural_lam.datasets.meps.weather_dataset import WeatherDataset
from neural_lam.datasets.mllam import GraphWeatherModelDataset
from neural_lam.models.graph_lam import GraphLAM
from neural_lam.utils import load_graph

FP_TESTDATA = Path(
    "/home/leif/git-repos/dmi/mllam/mllam-data-prep/example.danra.zarr"
)

# Third-party
import numpy as np
import weather_model_graphs as wmg
from loguru import logger


def test_train_with_mllam_dataset():
    n_prediction_timesteps = 19
    fp_data = FP_TESTDATA
    dataset = GraphWeatherModelDataset(
        dataset_path=fp_data, n_prediction_timesteps=n_prediction_timesteps
    )

    n_forcing_features = 2
    n_state_features = 12
    n_static_features = 1

    n_input_steps = dataset.n_input_timesteps
    n_grid = dataset.ds.grid_index.shape[0]
    n_forcing_features = dataset.ds.forcing_feature.shape[0]
    n_state_features = dataset.ds.state_feature.shape[0]
    n_static_features = dataset.ds.static_feature.shape[0]

    # check that the dataset is not empty
    assert len(dataset) > 0

    # get the first item
    item = dataset[0]

    # check that the item contains exactly the tensors for prev_state, target_state, forcing and static
    assert "init_states" in item
    assert "target_states" in item
    assert "forcing_windowed" in item
    assert "static_features" in item
    assert len(item) == 4

    # check that the shapes of the tensors are correct
    assert item["init_states"].shape == (2, n_grid, n_state_features)
    assert item["target_states"].shape == (
        n_prediction_timesteps,
        n_grid,
        n_state_features,
    )
    assert item["forcing_windowed"].shape == (
        n_prediction_timesteps,
        n_grid,
        n_forcing_features * (n_input_steps + 1),
    )
    assert item["static_features"].shape == (n_grid, n_static_features)

    xy_grid = dataset.xy_coords

    # Third-party
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        fp_graph = fp_data.parent / f"{fp_data.stem}.graph"
        _create_graph(xy_grid=xy_grid, fp_graph=fp_graph)

        model_graph = load_graph(graph_dir_path=fp_graph)

    model = GraphLAM()


def test_train_with_meps_dataset():
    dataset = WeatherDataset(dataset_name="meps_example")

    n_forcing_features = 5
    n_state_features = 17
    n_static_features = 1
    n_input_steps = 2
    n_prediction_timesteps = dataset.sample_length - n_input_steps

    grid_xy = dataset.grid_xy
    nx, ny = grid_xy.shape[1:]
    n_grid = nx * ny

    # check that the dataset is not empty
    assert len(dataset) > 0

    # get the first item
    item = dataset[0]

    # check that the item contains exactly the tensors for prev_state, target_state, forcing and static
    assert "init_states" in item
    assert "target_states" in item
    assert "forcing_windowed" in item
    assert "static_features" in item
    assert len(item) == 4

    # check that the shapes of the tensors are correct
    assert item["init_states"].shape == (2, n_grid, n_state_features)
    assert item["target_states"].shape == (
        n_prediction_timesteps,
        n_grid,
        n_state_features,
    )
    assert item["forcing_windowed"].shape == (
        n_prediction_timesteps,
        n_grid,
        n_forcing_features * (n_input_steps + 1),
    )
    assert item["static_features"].shape == (n_grid, n_static_features)

    dataset_props = dataset.get_props()
    
    required_props = {'border_mask', 'grid_static_features', 'step_diff_mean', 'step_diff_std', 'data_mean', 'data_std', 'param_weights'}
    
    # check the sizes of the props
    assert dataset_props["border_mask"].shape == (n_grid, 1)
    assert dataset_props["grid_static_features"].shape == (n_grid, n_static_features)
    assert dataset_props["step_diff_mean"].shape == (n_state_features,)
    assert dataset_props["step_diff_std"].shape == (n_state_features,)
    assert dataset_props["data_mean"].shape == (n_state_features,)
    assert dataset_props["data_std"].shape == (n_state_features,)
    assert dataset_props["param_weights"].shape == (n_state_features,)

    
    assert set(dataset_props.keys()) == required_props


def _create_graph(xy_grid, fp_graph):
    logger.info("starting graph creation")
    # create the full graph
    graph = wmg.create.archetype.create_keisler_graph(xy_grid=xy_grid)

    logger.info("splitting")
    # split the graph by component
    graph_components = wmg.split_graph_by_edge_attribute(
        graph=graph, attribute="component"
    )

    logger.info("saving")
    # save the graph components to disk in pytorch-geometric format
    for component_name, graph_component in graph_components.items():
        kwargs = {}
        if component_name == "m2m":
            kwargs["list_from_attribute"] = "level"
        wmg.save.to_pyg(
            graph=graph_component,
            name=component_name,
            output_directory=fp_graph,
            **kwargs,
        )