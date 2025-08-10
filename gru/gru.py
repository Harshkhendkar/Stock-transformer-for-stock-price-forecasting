
import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self, d_model, hidden_size=None, num_layers=1, dropout=0.0, bidirectional=False):
        """
        GRU-based encoder for temporal modeling.

        Args:
            d_model (int): Input and output feature dimension.
            hidden_size (int): Hidden state size for GRU. Defaults to d_model if None.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout rate.
            bidirectional (bool): Whether to use bidirectional GRU.
        """
        super(GRUEncoder, self).__init__()
        if hidden_size is None:
            hidden_size = d_model
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        self.output_projection = nn.Linear(hidden_size * (2 if bidirectional else 1), d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        output, _ = self.gru(x)  # output shape: [B, T, hidden_size * num_directions]
        output = self.output_projection(output)  # Project back to d_model
        return output

