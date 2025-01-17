import torch
import torch.nn as nn


class FrequencyDomainLoss(nn.Module):
    def __init__(self):
        super(FrequencyDomainLoss, self).__init__()

    def forward(self, x, y):
        x_fft = torch.fft.rfft(x, dim=-1)  # Real-to-complex FFT along time dimension
        y_fft = torch.fft.rfft(y, dim=-1)
        x_magnitude = torch.abs(x_fft)
        y_magnitude = torch.abs(y_fft)
        return ((x_magnitude - y_magnitude) ** 2).reshape(x_magnitude.shape[0], -1)


class MultiChannelSpectralTemporalLoss(nn.Module):
    def __init__(self, fft_sizes=[256, 512], hop_sizes=None, win_lengths=None, alpha=1.0, beta=1.0, gamma=1.0):
        """
        Args:
            fft_sizes (list): FFT sizes for multi-resolution STFT.
            hop_sizes (list): Hop sizes for STFT (default: fft_size // 4).
            win_lengths (list): Window lengths for STFT (default: fft_size).
            alpha, beta (float): Weights for individual stream losses.
            gamma (float): Weight for shared signal loss.
        """
        super(MultiChannelSpectralTemporalLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes if hop_sizes else [size // 4 for size in fft_sizes]
        self.win_lengths = win_lengths if win_lengths else fft_sizes
        self.alpha = alpha
        self.beta = beta

    def compute_stft_loss(self, x, y):
        stft_loss = 0.0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win_length, device=x.device)
            # Compute STFT along the last dimension (sequence length)
            x_stft = torch.stft(x, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)
            y_stft = torch.stft(y, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)
            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)
            stft_loss += torch.mean((x_mag - y_mag) ** 2, dim=(-2, -1))
        return stft_loss

    def forward(self, x, y):
        """
        Args:
            x (torch.Tensor): Reconstructed waveform (B, 2, T).
            y (torch.Tensor): Ground-truth waveform (B, 2, T).
        """
        # Individual stream losses
        loss_stream1 = self.compute_stft_loss(x[:, 0, :], y[:, 0, :])  # Hanford
        loss_stream2 = self.compute_stft_loss(x[:, 1, :], y[:, 1, :])  # Livingston
        
        # Total loss
        total_loss = self.alpha * loss_stream1 + self.beta * loss_stream2
        return total_loss


def weight_norm(net):
    norm = 0
    for param in net.parameters():
        norm += (param ** 2).sum()
    return norm

