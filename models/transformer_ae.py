import math
import torch
import torch.nn as nn
from timm.layers import trunc_normal_


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

        self.init_weight()
        self.apply(self._init_weights)
        
    def init_weight(self):
        if self.latent_token is not None:
            trunc_normal_(self.latent_token, std=.02)
        if self.pos_embedding is not None:
            trunc_normal_(self.pos_embedding, std=.02)
        
    def _init_weights(self, module):
        """
        Initializes the weights of the model layers.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            trunc_normal_(module.weight, std=.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
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

        self.init_weight()
        self.apply(self._init_weights)
        
    def init_weight(self):
        if self.pos_embedding is not None:
            trunc_normal_(self.pos_embedding, std=.02)
        
    def _init_weights(self, module):
        """
        Initializes the weights of the model layers.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            trunc_normal_(module.weight, std=.02)
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer normalization parameters
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
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


