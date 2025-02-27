import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=200):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, mlp_head: bool = None, in_features: int = 2, seq_len : int = 200, window_size: int = 1,
                 pos_embedding: bool = None, z_dim: int = 32, num_layers: int = 5, 
                 latent_token: bool = None, num_head: int = 8, dropout: float = 0.1):
        super(Encoder, self).__init__()
        
        self.seq_len = seq_len
        self.window_size = window_size
        self.pos_embedding = pos_embedding
        self.latent_token = latent_token
        self.mlp_head = mlp_head

        if mlp_head is not None:
            assert latent_token is not None
            self.mlp_head = nn.Linear(z_dim, 1)

        # Embedding for each waveform position
        #self.embedding = nn.Linear(window_size * in_features, z_dim)
        self.patchify = torch.nn.Conv1d(in_features, z_dim, kernel_size=window_size, stride=window_size)
        self.dropout = nn.Dropout(p=dropout)
 
        # Positional encoding to retain sequence information
        extra_token = 1 if latent_token is not None else 0
        if pos_embedding is not None:
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len // window_size + extra_token, z_dim))
        else:
            self.pos_encoding = PositionalEncoding(d_model=z_dim, max_len=seq_len // window_size + extra_token)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=z_dim,
            nhead=num_head,
            dim_feedforward=z_dim * 4,
            batch_first=True,
            dropout=dropout
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Latent-space token (additional to the sequence)
        if latent_token is not None:
            self.latent_token = nn.Parameter(torch.zeros(1, 1, z_dim))
        
        # Token type embedding to distinguish between latent token and sequence tokens
        self.token_type = torch.tensor([0, 1]).view(1, 1, 2).long()
        self.token_embedding = nn.Embedding(2, z_dim)

    def forward(self, x):
        # Ensure token_type is on the same device as the input x
        device = x.device
        token_type = self.token_type.to(device)
        batch_size = x.size(0)
 
        # Patchify
        x = self.patchify(x)
        x = torch.einsum('bfs -> bsf', x)
        if self.latent_token is not None:
            # Include the classification token at the beginning of the sequence
            latent_tokens = self.latent_token.repeat(x.size(0), 1, 1)
            x = torch.cat((latent_tokens, x), dim=1)
            
            # Add token type embeddings
            x[:, :1, :] += self.token_embedding(token_type[:, :, 0])
            x[:, 1:, :] += self.token_embedding(token_type[:, :, 1])
        
        # Add positional encoding
        x = x + self.pos_embedding if self.pos_embedding is not None else self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through the encoder
        z = self.encoder(x)[:, 0] if self.latent_token is not None else self.encoder(x)
        
        if self.mlp_head is not None:
            return self.mlp_head(self.dropout(z))

        return z


class Decoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, in_features: int = 2, seq_len : int = 200, window_size: int = 1,
                 pos_embedding: bool = None, z_dim: int = 32, num_layers: int = 5, 
                 latent_token: bool = None, num_head: int = 8, dropout: float = 0.1):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.window_size = window_size
        self.z_dim = z_dim
        self.pos_embedding = pos_embedding
        self.latent_token = latent_token
        
        if latent_token is not None:
            self.latent_to_seq = nn.Linear(z_dim, seq_len // window_size * z_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Positional encoding to retain sequence information
        if pos_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len // window_size, z_dim))
        else:
            self.pos_encoding = PositionalEncoding(d_model=z_dim, max_len=seq_len // window_size)

        # Transformer encoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=z_dim,
            nhead=num_head,
            dim_feedforward=z_dim * 4,
            batch_first=True,
            dropout=dropout
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Fully connected layers
        self.final_mlp = nn.Linear(z_dim, in_features * window_size)

    def forward(self, z):       
        batch_size = z.size(0)
        
        # Repeat
        #z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        if self.latent_token is not None:
            z = self.latent_to_seq(z).view(-1, self.seq_len // self.window_size, self.z_dim)
            z = self.dropout(z)

        # Add positional encoding
        #z = z + self.pos_embedding if self.pos_embedding is not None else self.pos_encoding(x)
        
        # Pass through the decoder
        emb = self.decoder(z)
        emb = self.dropout(emb)
        x_hat = self.final_mlp(emb)
        
        # Reshape back to seq_len
        x_hat = x_hat.view(batch_size, -1, self.seq_len)
        
        return x_hat


class Model(nn.Module):
    """autoencoder class"""

    def __init__(
        self,
    ):
        super().__init__()

    def load(self):
        args = {'in_features': 1,
            'seq_len': 100,
            'window_size': 10,
            'z_dim': 128,
            'num_layers': 6,
            'num_head': 8,
            'dropout': 0.1,
            'latent_token': True,
            'pos_embedding': True} 
        self.encoder = Encoder(**args)
        self.decoder = Decoder(**args)
        self.spherical = True
        self.encoding_noise = None
        self.eps = 1e-8
        self.loss = "l2"
        self.learn_out_scale = True
        if self.learn_out_scale is not None:
            self.register_parameter("out_scale", nn.Parameter(torch.tensor(1.0)))
        weight_path = os.path.join(os.path.dirname(__file__), 'model_weights.pt')
        self.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        self.eval()


    def encode(self, x, noise=False, train=False):
        """wrapper for encoder"""
        z = self.encoder(x)
        z_pre = z
        if noise and self.encoding_noise is not None:
            if self.spherical:
                z = self._project(z)
            z = z + torch.randn_like(z) * self.encoding_noise
        if self.spherical:
            z = self._project(z)

        if train:
            return {"z": z, "z_pre": z_pre}
        else:
            return z

    def decode(self, x):
        """wrapper for decoder"""
        return self.decoder(x)

    def _project(self, z):
        """project to unit sphere"""
        return z / (torch.norm(z, dim=1, keepdim=True) + self.eps)

    def recon_error(self, x, noise=False):
        """reconstruction error"""
        z = self.encode(x, noise=noise)
        x_recon = self.decode(z)
        self.z = z
        if self.loss == "l2":
            return ((x - x_recon) ** 2).view(len(x), -1).mean(dim=1)
        elif self.loss == "l2_sum":
            return ((x - x_recon) ** 2).view(len(x), -1).sum(dim=1)
        elif self.loss == "l1":
            return (torch.abs(x - x_recon)).view(len(x), -1).mean(dim=1)

    def forward(self, x):
        recon = self.recon_error(x, noise=False)
        self.state = {"recon": recon}

        if self.learn_out_scale == 'exp':
            return recon * (torch.exp(self.out_scale))
        elif self.learn_out_scale is not None:  # backward compatibility
            return recon * ((self.out_scale)**2)
        else:
            return recon

    def fft(self, data):
        """
        Computes the cross-power spectrum magnitude between the Hanford and Livingston LIGO detectors.
        """
        fft_hanford = torch.fft.rfft(data[:, 0, :], dim=1)     # Hanford channel
        fft_livingston = torch.fft.rfft(data[:, 1, :], dim=1)  # Livingston channel

        cross_spectrum = fft_hanford * torch.conj(fft_livingston)
        cross_magnitude = torch.abs(cross_spectrum)
    
        return cross_magnitude[:, :100].reshape(-1, 1, 100)
 
    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        # Transpose only if the last two dimensions are (200, 2)
        if x.shape[-2:] == (200, 2):
            x = x.transpose(-1, -2)

        x = self.fft(x) / 1000.
        
        preds = self(x).detach().cpu().numpy()
        
        return preds

