# Slot: ffn (v2, by Nick Grebe)

class FFN(nn.Module):
    """Parameter-efficient SwiGLU feed-forward network."""

    def __init__(self, cfg):
        super().__init__()
        #Approximately matches the parameter count of a standard 4x FFN:
        hidden_dim = int((8 / 3) * cfg.n_embd)

        # Hardware-friendly rounding.
        hidden_dim = 64 * ((hidden_dim + 63) // 64)

        self.gate = nn.Linear(
            cfg.n_embd,
            hidden_dim,
            bias=False,
        )
        self.up = nn.Linear(
            cfg.n_embd,
            hidden_dim,
            bias=False,
        )
        self.down = nn.Linear(
            hidden_dim,
            cfg.n_embd,
            bias=False,
        )

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = F.silu(self.gate(x)) * self.up(x)
        return self.drop(self.down(x))
