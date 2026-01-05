import math
import torch
import torch.nn as nn
from typing import Dict


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

class MessiTransformer(nn.Module):
    def __init__(self, cat_dims: Dict, cont_dim=5, d_model=128, nhead=4,
                 num_layers=2, emb_dim=16, dropout=0.1, max_len=128):
        super().__init__()

        # Feature embedding & projection
        self.embeddings = nn.ModuleDict()
        self.cat_names = list(cat_dims.keys())

        for name, num_classes in cat_dims.items():
            self.embeddings[name] = nn.Embedding(num_classes + 1, emb_dim, padding_idx=0)

        total_input_dim = cont_dim + len(cat_dims) * emb_dim

        self.input_projection = nn.Sequential(
            nn.Linear(total_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Transformer backbone
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, cont_x, cat_x, padding_mask=None):

        batch_size, seq_len, _ = cont_x.shape

        # categorical feature embedding
        embedded_cat = []
        for i, name in enumerate(self.cat_names):
            col_data = cat_x[:, :, i]
            emb = self.embeddings[name](col_data)
            embedded_cat.append(emb)

        cat_features = torch.cat(embedded_cat, dim=-1)

        # concatenate continuous and categorical features
        combined_x = torch.cat([cont_x, cat_features], dim=-1)

        # input projection
        x = self.input_projection(combined_x)

        # positional encoding
        x = self.pos_encoder(x)

        # transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # pooling and prediction
        last_token_output = x[:, -1, :]
        prediction = self.output_head(last_token_output)
        return prediction