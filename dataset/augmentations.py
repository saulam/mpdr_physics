import numpy as np
from scipy.signal import butter, lfilter, resample


def add_noise(data, noise_level=0.05):
    """
    Add Gaussian noise to the data.
    """
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def time_shift(data, max_shift=10):
    """
    Shift the data in time by a specified number of samples.
    """
    shift = np.random.randint(low=-max_shift, high=max_shift+1)
    return np.roll(data, shift, axis=-1)

def time_stretch(data, stretch_range=(0.95, 1.05)):
    """
    Stretch or compress the time dimension of the data.
    The stretch factor is sampled randomly from the specified range.
    The waveform is adjusted to maintain the original length by wrapping.
    """
    stretch_factor = np.random.uniform(*stretch_range)
    num_samples = int(data.shape[1] * stretch_factor)
    stretched = np.array([resample(data[f], num_samples) for f in range(data.shape[0])])
    
    if num_samples > data.shape[1]:  # crop
        adjusted = stretched[:, :data.shape[1]]
    else:  # wrap to maintain length
        adjusted = np.zeros_like(data)
        for f in range(data.shape[0]):
            adjusted[f, :num_samples] = stretched[f]
            adjusted[f, num_samples:] = stretched[f, :data.shape[1] - num_samples]
    
    return adjusted

def amplitude_scaling(data, scale_range=(0.95, 1.05)):
    """
    Scale the amplitude of the data randomly within the specified range.
    """
    scale = np.random.uniform(*scale_range)
    return data * scale

