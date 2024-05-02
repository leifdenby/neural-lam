# First-party
from neural_lam.datasets.meps.weather_dataset import WeatherDataset
from neural_lam.datasets.mllam import GraphWeatherModelDataset

FP_TESTDATA = "/home/lcd/git-repos/mllam-data-prep/example.danra.zarr"


def test_train_with_mllam_dataset():
    n_prediction_timesteps = 19
    dataset = GraphWeatherModelDataset(
        dataset_path=FP_TESTDATA, n_prediction_timesteps=n_prediction_timesteps
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
