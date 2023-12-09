


from typing import Optional
from torchvision import transforms
from torch.utils.data import random_split
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pytorch_lightning as pl



######################################## 
# PREPROCESSING

def coords_zero(*pos):
    """
    coords at zero are not valid;
    this function returns True if all coords are zero, False otherwise
    """
    return sum(pos) == 0
    

def read_data(data:pd.DataFrame, trajectory_count:int) -> list:

    """
    Preprocesses the training data.

    Parameters
    ----------

    `data` - pd.DataFrame:
        The training data.

    `trajectory_count` - int:
        The number of time steps in each trajectory.

    Returns
    -------
    `data` - list:
        Training data.
    """
    
    data = data[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 't']]
    # data = data[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3', 'v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3']]
    data = data.to_numpy()
    num_trajectories = int(len(data) / trajectory_count)
    data = [data[trajectory_count*traj_i:int(trajectory_count)*(traj_i+1)] for traj_i in range(num_trajectories)] # shape: (num_trajectories, trajectory_count, 6)

    # Filter out trajectories where all positions are the same (collision)
    for i in range(num_trajectories):
        
        trajectory = data[i] # shape: (trajectory_count, 6)
        coords = np.vectorize(coords_zero)(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], trajectory[:, 4], trajectory[:, 5]) # shape: (trajectory_count)
        data[i] = trajectory[~coords] # shape: (trajectory_count, 6)
        # print(f"Size {len(trajectory)} -> {len(data[i])}")
    

    return data


def preprocess_data(data:list):
    # shape of data is (num_trajectories, trajectory_count, 6)

    stand_data = np.concatenate(data, axis=0)
    means = np.mean(stand_data, axis=0)
    stddev = np.std(stand_data, axis=0)

    # standardize data
    data = [(trajectory - means) / stddev for trajectory in data] # shape: (trajectory_count, N_time_steps, feat_size)

    return data, means, stddev




def make_windows(data, window_size):
    """
    Given a list of trajectories, each trajectory is a numpy array of shape (trajectory_count, 6)
    Return a list of windows of shape (window_size, 6)
    """
    windows = []
    for trajectory in data:
        for i in range(len(trajectory) - window_size):
            windows.append(trajectory[i:i+window_size])
    return windows



class TrajectoryDataset(Dataset):
    
    def __init__(self, data_path, window_size, test_split=0.2) -> None:
        
        self.orbit_data = read_data(pd.read_csv(data_path), trajectory_count=257) # shape = (num_trajectories, trajectory_count, 6)
        
        # filter tranjectories with less than window size time steps
        self.orbit_data = [trajectory for trajectory in self.orbit_data if len(trajectory) >= window_size + 1] # shape = (num_trajectories, trajectory_count, 6)

        print(f"Number of trajectories in dataset: {len(self.orbit_data)}")

        # split train and test
        self.orbit_data, self.test_data = random_split(self.orbit_data, [1 - test_split, test_split])

        # get time steps
        time_steps = [trajectory[:, -1] for trajectory in self.orbit_data]
        # normalize time steps
        time_steps = [(time_step - np.min(time_step)) / (np.max(time_step) - np.min(time_step)) for time_step in time_steps]

        # preprocess data
        self.orbit_data, self.means, self.stddev = preprocess_data(self.orbit_data)
        # replace time steps with the original ones
        self.orbit_data = [np.concatenate((trajectory[:, :-1], time_steps[i].reshape(-1, 1)), axis=1) for i, trajectory in enumerate(self.orbit_data)]

        # preprocess test data
        time_steps = [trajectory[:, -1] for trajectory in self.test_data]
        time_steps = [(time_step - np.min(time_step)) / (np.max(time_step) - np.min(time_step)) for time_step in time_steps]

        self.test_data = [(trajectory - self.means) / self.stddev for trajectory in self.test_data]
        self.test_data = [np.concatenate((trajectory[:, :-1], time_steps[i].reshape(-1, 1)), axis=1) for i, trajectory in enumerate(self.test_data)]

        print(f"Number of trajectories in train dataset: {len(self.orbit_data)}")
        print(f"Number of trajectories in test dataset: {len(self.test_data)}")

        # make windows
        self.train_windows = make_windows(self.orbit_data, window_size + 1) # shape = (num_windows, window_size, 6)
        
    def __len__(self):
        return len(self.train_windows)
    
    def __getitem__(self, idx):
        wind_idx = torch.tensor(self.train_windows[idx], dtype=torch.float32) # shape: (window_size +1, feat_size), where the last idx is is the t+1 time step
        # print(wind_idx.shape) # (window_size + 1, feat_size + 1); +1 on dim_0 is the gt time sample and the +1 on the dim_1 is the time feature
        return wind_idx[:-1], wind_idx[-1]
    

    def get_test_trajectories(self):
        return self.test_data
    
    def get_feat_size(self):
        return self.orbit_data.shape[-1]
        

class OrbitDataModule(pl.LightningDataModule):

    def __init__(self, 
                 data_dir: str = "./", 
                 window_size = 10,
                 batch_size: int = 32, 
                 num_workers: int = 12):
        
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.window_size = window_size

    def setup(self, stage: Optional[str] = None):

        orbit_full = TrajectoryDataset(self.data_dir, self.window_size)

        if stage == "fit" or stage == 'train' or stage == 'val':
            self.orbit_train, self.orbit_val = random_split(orbit_full, [0.8, 0.2])
        elif stage == "test":
            self.orbit_test = orbit_full.get_test_trajectories()

    def train_dataloader(self):
        return DataLoader(self.orbit_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.orbit_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        # During test time, the data is no longer split into windows, insteads the whole trajectory is used
        # thus, the network takes as input the first time step and predicts the next one, so on and so forth
        # the network is evaluated on the whole trajectory
        return DataLoader(self.orbit_test, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.orbit_test, batch_size=1, num_workers=self.num_workers)
    
    def get_standardization_params(self):
        # can be used during predict time to recover the original trajectory values
        return self.orbit_train.means, self.orbit_train.stddev





if __name__ == "__main__":

    import sys
    sys.path.insert(0, '..')
    sys.path.insert(1, '../..')
    
    from my_utils.constants import ORBIT_DATASET_PATH

    
    traj_data = TrajectoryDataset(ORBIT_DATASET_PATH, window_size=10)


    print(traj_data[0][0].shape, traj_data[0][1].shape) # torch.Size([10, 6]) torch.Size([6])
    print(traj_data[0][0])
    