# This file contains the custom (non-HF) modules used in the model architectures.
import os
import torch
import torch.nn as nn
from typing import List
from experiment.parameters import ExperimentParams
import names


class FeatureDropout(nn.Module):
    def __init__(self, p: float):
        super(FeatureDropout, self).__init__()
        self.dropout = nn.Dropout1d(p)

    def forward(self, x):
        if len(x.shape) == 3:
            # batch, time, features
            return self.dropout(x.transpose(1, 2)).transpose(1, 2)
        else:
            # batch, features
            return self.dropout(x.unsqueeze(-1)).squeeze(-1)


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(size), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.gamma * (x * rms).to(input_dtype)


class SwiGLU(nn.Module):
    # https://arxiv.org/pdf/2002.05202v1.pdf
    def __init__(self, size):
        super().__init__()
        self.W = nn.Linear(size, size, bias=False)
        self.V = nn.Linear(size, size, bias=False)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        swish = self.W(x) * torch.sigmoid(self.beta * self.W(x))
        return swish * self.V(x)


def create_layers(
    input_dim: int,
    sizes: List[int] = None,
    dropout_p: float = 0.0,
    activation: str = "swiglu",
):
    layers = []
    for size in sizes:
        if size > 0:
            layers += [
                nn.Linear(input_dim, size, bias=False),
                SwiGLU(size) if activation == "swiglu" else nn.ReLU(),
                FeatureDropout(dropout_p) if dropout_p > 0 else nn.Identity(),
                RMSNorm(size),
            ]
            input_dim = size
    return layers


class MLPModel(nn.Module):
    # Dropout is dropout over time
    def __init__(
        self, sequence_length, hidden_size, num_layers, dropout_p=0.0, activation="relu"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        if num_layers == 0:
            layers = [nn.Identity()]
        else:
            sizes = [sequence_length for _ in range(num_layers)]
            layers = create_layers(
                input_dim=sequence_length,
                sizes=sizes,
                dropout_p=dropout_p,
                activation=activation,
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # Transpose to (batch, hidden_size, num_frames). MLP over time
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(2, 1)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p, attention_dropout_p=0.0):
        super(TransformerBlock, self).__init__()
        self.mha = MultiheadAttention(embed_dim, num_heads, attention_dropout_p)
        self.rms = RMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            *create_layers(
                embed_dim, sizes=[embed_dim], dropout_p=dropout_p, activation="swiglu"
            )
        )

    def forward(self, x):
        # Attention
        x = x + self.mha(x)
        x = self.rms(x)

        # MLP
        x = x + self.mlp(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attention_dropout_p=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.scale = 1 / (self.head_dim**0.5)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = None
        if attention_dropout_p > 0:
            self.dropout = nn.Dropout(attention_dropout_p)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        q = self.query(x)
        k = self.key(x) * self.scale
        v = self.value(x)

        qk = torch.einsum(
            "bthd,bshd->bhts",
            q.view(batch_size, seq_length, self.num_heads, self.head_dim),
            k.view(batch_size, seq_length, self.num_heads, self.head_dim),
        )
        mask = torch.tril(
            torch.ones(seq_length, seq_length, device=x.device), diagonal=0
        )
        # Non-linearity over time
        qk = qk.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(qk, dim=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        # attn_weights = self.dropout(attn_weights)
        v = torch.unflatten(v, dim=2, sizes=[-1, self.num_heads])
        qkv = torch.einsum("kjlm,kmij->kiml", v, attn_weights)
        qkv = torch.flatten(qkv, start_dim=2, end_dim=3)
        qkv = self.layer_norm(qkv)
        return self.out_proj(qkv)


class TransformerModel(nn.Module):
    def __init__(
        self, hidden_size, num_heads, num_layers, dropout_p, attention_dropout_p=0.0
    ):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, dropout_p, attention_dropout_p)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @classmethod
    def from_pretrained(cls, sub_folder):
        params = ExperimentParams.load_from_subfolder(sub_folder)
        config = params.sequence_config.config
        model = cls(**config)
        pretrained_path = os.path.join(
            names.MODELS_DIR, sub_folder, "SequenceModel", "best_model.pt"
        )
        model.load_state_dict(
            torch.load(pretrained_path, map_location=torch.device("cpu"))
        )
        return model


class LSTMModel(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout_p):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p,
            batch_first=True,
        )

    def forward(self, x):
        # LSTM expects input of shape (batch, seq_len, input_size)
        # Output shape of LSTM: (batch, seq_len, hidden_size)
        output, (hidden_state, cell_state) = self.lstm(x)
        # We return the last hidden state
        return hidden_state[-1].unsqueeze(1)

    @classmethod
    def from_pretrained(cls, sub_folder):
        params = ExperimentParams.load_from_subfolder(sub_folder)
        config = params.sequence_config.config
        model = cls(**config)
        pretrained_path = os.path.join(
            names.MODELS_DIR, sub_folder, "SequenceModel", "best_model.pt"
        )
        model.load_state_dict(
            torch.load(pretrained_path, map_location=torch.device("cpu"))
        )
        return model
