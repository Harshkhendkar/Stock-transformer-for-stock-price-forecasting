
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]

class AutoCorrelation(nn.Module):
    def __init__(self, factor=0.5):
        super().__init__()
        self.factor = factor

    def forward(self, q, k, v):
        # Fourier-based autocorrelation: simplified
        B, T, C = q.shape
        fft_q = torch.fft.rfft(q, dim=1)
        fft_k = torch.fft.rfft(k, dim=1)
        corr = torch.fft.irfft(fft_q * torch.conj(fft_k), n=T, dim=1)
        weights = torch.softmax(corr, dim=1)
        output = torch.einsum("btd,btv->btd", weights, v)
        return output

class AutoInformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.value_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.corr = AutoCorrelation()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        attn_output = self.corr(q, k, v)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.ffn(x))
        return x


