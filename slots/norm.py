# Slot: norm (v4, by Shubh Goyal)

class Norm(nn.Module):
    """RMSNorm — cheaper than LayerNorm (no mean-subtraction, no bias),
    and typically gives slightly better loss at this scale."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight
