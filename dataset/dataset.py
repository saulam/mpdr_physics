import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from dataset.augmentations import *


class GWDataset(Dataset):
    def __init__(self, file_path, augment=False):
        """
        Args:
            file_path (str): Path to the NumPy file containing the dataset (shape: (N, 2, 200)).
        """
        loaded_data = np.load(file_path)
        self.training = False
        self.augmentations = augment

        if isinstance(loaded_data, np.lib.npyio.NpzFile) and 'data' in loaded_data:
            self.data = loaded_data['data']
        else:
            self.data = loaded_data

        stds = np.std(self.data, axis=-1)[:, :, np.newaxis]
        self.data = self.data/stds

    def __len__(self):
        """
        Returns the number of examples in the dataset.
        """
        return self.data.shape[0]

    def set_training_mode(self, training=True):
        """Sets the split type dynamically."""
        #print("Setting training mode to {}.".format(training))
        self.training = training

    def augment(self, x):
        x = add_noise(x)
        x = time_shift(x)
        x = time_stretch(x)
        #x = amplitude_scaling(x)
 
        return x


    def clamp_and_normalize(self, waveform, lower_percentile=2.5, upper_percentile=97.5):
        """
        Clamps outliers and normalizes a single waveform to the range [-1, 1].
        """
        lower_bound = np.percentile(waveform, lower_percentile, axis=0, keepdims=True)
        upper_bound = np.percentile(waveform, upper_percentile, axis=0, keepdims=True)
        waveform_clamped = np.clip(waveform, lower_bound, upper_bound)
        min_vals = np.min(waveform_clamped, axis=0, keepdims=True)
        max_vals = np.max(waveform_clamped, axis=0, keepdims=True)
        normalized_waveform = 2 * (waveform_clamped - min_vals) / (max_vals - min_vals + 1e-10) - 1

        return normalized_waveform


    def fft(self, data):
        """
        Computes the cross-power spectrum magnitude between the Hanford and Livingston LIGO detectors.
        """
        fft_hanford = torch.fft.rfft(data[0, :], dim=0)     # Hanford channel
        fft_livingston = torch.fft.rfft(data[1, :], dim=0)  # Livingston channel

        cross_spectrum = fft_hanford * torch.conj(fft_livingston)
        cross_magnitude = torch.abs(cross_spectrum)
    
        return cross_magnitude


    def __getitem__(self, idx):
        """
        Retrieve one example from the dataset, permute dimensions 1 and 2, 
        and convert the NumPy array to a PyTorch tensor.

        Args:
            idx (int): Index of the example to retrieve.

        Returns:
            torch.Tensor: The transformed example.
        """
        example = self.data[idx]
        if self.training and self.augmentations:
            if np.random.rand() > 0.01:
                example = self.augment(example)
        #example = self.clamp_and_normalize(example)
        example_tensor = torch.tensor(example, dtype=torch.float32)
        example_tensor_fft = self.fft(example_tensor)[:100] / 1000.
        example_tensor_fft = example_tensor_fft.reshape(1, -1)

        return example_tensor_fft


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """
    Split a dataset into training and test sets.

    Args:
        dataset (Dataset): The dataset to split.
        train_ratio (float): Proportion of the dataset to use for training (default is 0.8).

    Returns:
        Tuple[Dataset, Dataset]: Training and test datasets.
    """
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(seed))
    return train_dataset, test_dataset


def create_data_loader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the NumpyDataset.

    Args:
        file_path (str): Path to the NumPy file.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=shuffle, pin_memory=True, num_workers=num_workers, 
                        persistent_workers=True if num_workers > 0 else False)
    return loader

