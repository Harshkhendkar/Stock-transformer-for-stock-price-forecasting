
import torch
import torch.nn as nn

class ProbSparseSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.dim_per_head = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        B, L, D = queries.shape
        H = self.nhead

        q = self.q_proj(queries).view(B, L, H, -1).transpose(1, 2)  # B, H, L, D_head
        k = self.k_proj(keys).view(B, L, H, -1).transpose(1, 2)
        v = self.v_proj(values).view(B, L, H, -1).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.dim_per_head ** 0.5
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(context)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseSelfAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        new_x = self.self_attn(x, x, x)
        x = x + new_x
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

class InformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, layers=1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dropout) for _ in range(layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

