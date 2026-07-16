# Slot: architecture (v0, by Deep-ML)

class Block(nn.Module):
    """Pre-norm transformer block (vanilla nanoGPT wiring)."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = Norm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = Norm(cfg.n_embd)
        self.ffn = FFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    """Embeddings -> N blocks -> final norm -> untied lm_head."""

    def __init__(self, cfg):
        super().__init__()
        self.embeddings = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = Norm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx):
        x = self.embeddings(idx)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.norm_f(x))


def build_model(cfg):
    """The topology is yours: rewire blocks, tie weights, go parallel."""
    return GPT(cfg)
