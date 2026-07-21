# Slot: ffn (v7, by anonymous)

class FFN(nn.Module):
    """Parameter-efficient SwiGLU with a fused gate/up projection."""

    def __init__(self, cfg):
        super().__init__()
        hidden_dim = int((8 / 3) * cfg.n_embd)
        hidden_dim = 64 * ((hidden_dim + 63) // 64)

        self.gate_up = nn.Linear(
            cfg.n_embd,
            2 * hidden_dim,
            bias=False,
        )

        self.down = nn.Linear(
            hidden_dim,
            cfg.n_embd,
            bias=False,
        )

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        return self.drop(self.down(x))
