# Slot: ffn (v0, by Deep-ML)

class FFN(nn.Module):
    """Position-wise feed-forward, 4x expansion + GELU (vanilla nanoGPT)."""

    def __init__(self, cfg):
        super().__init__()
        self.up = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.down = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))
