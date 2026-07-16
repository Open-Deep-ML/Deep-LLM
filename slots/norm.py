# Slot: norm (v0, by Deep-ML)

class Norm(nn.Module):
    """Standard LayerNorm."""

    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x)
