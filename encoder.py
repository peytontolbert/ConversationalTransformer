import torch.nn as nn

# Unified Transformer Encoder with Liquid Time-Constant Networks
class LiquidTransformerEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(LiquidTransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.ltc_layers = nn.ModuleList([
            LiquidTimeConstantLayer(hidden_size) for _ in range(6)
        ])

    def forward(self, src, src_mask=None):
        output = src
        for layer in self.ltc_layers:
            output = layer(output, src_mask)
        return output

# Liquid Time-Constant Layer
class LiquidTimeConstantLayer(nn.Module):
    def __init__(self, hidden_size):
        super(LiquidTimeConstantLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x, mask=None):
        # Simulate continuous-time dynamics
        delta_t = 0.1  # Time step
        h_prev = x
        A = self.W(x)
        B = self.U(x)
        h = h_prev + delta_t * (-A * h_prev + B * self.activation(h_prev))
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        return h
