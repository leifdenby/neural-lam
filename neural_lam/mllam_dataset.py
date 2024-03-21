import os
import glob
import torch
import numpy as np
import datetime as dt
import random
import xarray
from copy import deepcopy

from neural_lam import utils, constants


class MllamDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):

        return init_states, target_states, static_features, forcing_windowed


class GraphWeatherModelDataset(MllamDataset):
    MODEL_INPUTS = dict(
        state=["time", "grid_index", "state_feature"],
        static=["grid_index", "feature"],
        forcing= ["time", "grid_index", "forcing_feature"]
    )




class AnalysisDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_name,
        pred_length=12,
        split="trainval",
        standardize=True,
        input_file=None,
    ):
        super().__init__()

        assert split in ("train", "val", "trainval", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join("data", dataset_name, "samples", split)
        self.static_dir_path = os.path.join("data", dataset_name, "static")
        self.sample_length = pred_length + 2  # 2 init states

        if input_file is None:
            zarr_files = glob.glob(os.path.join(self.sample_dir_path, "*.zarr"))

            assert len(zarr_files) > 0, "No samples found from {}".format(
                self.sample_dir_path
            )

            assert len(zarr_files) == 1, "Only one zarr file per directory supported"

            input_file = zarr_files[0]
        else:
            _, extension = os.path.splitext(input_file)

            assert extension == ".zarr", "Only zarr files supported"

        self.initialize_from_zarr(input_file)

        # Set up for standardization
        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean, self.data_std = (
                ds_stats["data_mean"],
                ds_stats["data_std"],
            )

        random.shuffle(self.samples)

    #    def initialize_from_npy(self):
    #        file_regexp = "analysis-*.npy"
    #        sample_paths = glob.glob(os.path.join(self.sample_dir_path, file_regexp))
    #
    #        assert len(sample_paths) > 0, "No samples found from {}".format(
    #            self.sample_dir_path
    #        )
    #
    #        self.toc = {}
    #
    #        for path in sample_paths:
    #            date = dt.datetime.strptime(path.split("/")[-1][9:-4], "%Y%m%d")
    #            for i in range(24):
    #                self.toc[date + dt.timedelta(hours=i)] = {"path": path, "index": i}
    #
    #        self.samples = []
    #        sample = []
    #        for k, v in self.toc.items():
    #            sample.append({k: v})
    #
    #            if len(sample) == self.sample_length:
    #                self.samples.append(sample)
    #                sample = []

    def initialize_from_zarr(self, filename):
        if filename.startswith("s3://"):
            import s3fs

            s3 = s3fs.S3FileSystem(anon=True, endpoint_url="https://lake.fmi.fi")
            store = s3fs.S3Map(root=filename, s3=s3, check=False)
            self.ds = xarray.open_zarr(store=store)
        else:
            self.ds = xarray.open_zarr(filename)

        times = self.ds["time"].values

        self.samples = []
        prev = 0

        for i in range(self.sample_length, len(times) + 1, self.sample_length):
            self.samples.append({"times": times[prev:i], "start": prev, "stop": i})
            prev = i

        assert (
            len(self.samples) > 0
        ), "No samples found from {}, required sample length: {} data length: {}".format(
            filename, self.sample_length, len(times)
        )
        print("Dataset initialized, length: {}".format(len(self.samples)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # === Sample ===
        #        def read_item(item):
        #            vv = next(iter(item.values()))
        #            data = np.load(vv["path"])[vv["index"]]
        #            return data

        sample = self.samples[idx]

        assert self.ds is not None, "dataset not initialized"

        sample_times = sample["times"]
        sample_times = [
            dt.datetime.strptime(str(t), "%Y-%m-%dT%H:%M:%S.000000000")
            for t in sample_times
        ]
        sample = self.ds.data[sample["start"] : sample["stop"]]
        sample = sample.to_numpy()
        # (N_t, N_x, N_y, d_features')
        sample = torch.tensor(sample, dtype=torch.float32).permute(0, 2, 3, 1)

        _, N_x, N_y, _ = sample.shape
        N_grid = N_x * N_y

        # Flatten spatial dim
        sample = sample.flatten(1, 2)  # (N_t, N_grid, d_features)

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean) / self.data_std

        # Split up sample in init. states and target states
        init_states = sample[:2]  # (2, N_grid, d_features)
        target_states = sample[2:]  # (sample_length-2, N_grid, d_features)

        # === Static batch features ===
        # Just a placeholder
        static_features = torch.zeros((N_grid, 1))

        # === Forcing features ===
        # Forcing features

        # Sun elevation angle
        sun_path = os.path.join(self.static_dir_path, f"sun_angle.npy")

        # Datetime used is the time of the forecast hour
        dt_obj = sample_times[0] + dt.timedelta(hours=2)
        start_of_year = dt.datetime(dt_obj.year, 1, 1)
        hour_into_year = int((dt_obj - start_of_year).total_seconds() / 3600)

        sun_angle = np.load(sun_path)[hour_into_year : hour_into_year + sample.shape[0]]

        # if rolling over to next year, add first hours
        if hour_into_year + sample.shape[0] > 8760:
            leftover = hour_into_year + sample.shape[0] - 8760
            sun_angle = np.concatenate((sun_angle, np.load(sun_path)[0:leftover]))

        assert sun_angle.shape[0] == sample.shape[0]

        sun_angle = (
            torch.Tensor(sun_angle.astype(np.float32))
            .flatten(1, 2)
            .unsqueeze(0)
            .permute(1, 2, 0)
        )

        # Extract for initial step
        init_hour_in_day = dt_obj.hour
        start_of_year = dt.datetime(dt_obj.year, 1, 1)
        init_seconds_into_year = (dt_obj - start_of_year).total_seconds()

        hour_of_day = torch.FloatTensor([int(x.strftime("%H")) for x in sample_times])
        second_into_year = torch.FloatTensor(
            [
                init_seconds_into_year + x * 3600
                for x in torch.arange(self.sample_length)
            ]
        )

        # Encode as sin/cos
        hour_angle = (hour_of_day / 12) * torch.pi  # (sample_len,)
        year_angle = (
            (second_into_year / constants.seconds_in_year) * 2 * torch.pi
        )  # (sample_len,)
        datetime_forcing = torch.stack(
            (
                torch.sin(hour_angle),
                torch.cos(hour_angle),
                torch.sin(year_angle),
                torch.cos(year_angle),
            ),
            dim=1,
        )  # (N_t, 4)
        datetime_forcing = (datetime_forcing + 1) / 2  # Rescale to [0,1]
        datetime_forcing = datetime_forcing.unsqueeze(1).expand(
            -1, N_grid, -1
        )  # (sample_len, N_grid, 4)

        # Put forcing features together
        forcing_features = torch.cat(
            (datetime_forcing, sun_angle), dim=-1
        )  # (sample_len, N_grid, d_forcing)

        # Combine forcing over each window of 3 time steps
        forcing_windowed = torch.cat(
            (
                forcing_features[:-2],
                forcing_features[1:-1],
                forcing_features[2:],
            ),
            dim=2,
        )  # (sample_len-2, N_grid, 3*d_forcing)
        # Now index 0 of ^ corresponds to forcing at index 0-2 of sample

        return init_states, target_states, static_features, forcing_windowed

    def split(self, train_ratio, val_ratio, test_ratio):
        """
        Splits the dataset into three datasets, train, val and test.
        """

        assert train_ratio + val_ratio + test_ratio == 1.0

        n = len(self)

        t_s, t_e = 0, int(train_ratio * n)
        v_s, v_e = t_e, int((train_ratio + val_ratio) * n)
        te_s, te_e = v_e, n

        train_ds = deepcopy(self)
        train_ds.samples = train_ds.samples[t_s:t_e]

        val_ds = deepcopy(self)
        val_ds.samples = val_ds.samples[v_s:v_e]

        test_ds = None

        if test_ratio > 0:
            test_ds = deepcopy(self)
            test_ds.samples = test_ds.samples[te_s:te_e]

        return train_ds, val_ds, test_ds

    def get_times(self):
        return [list(x["times"]) for x in self.samples]

    def data(self):
        return self.ds.data.to_numpy()
