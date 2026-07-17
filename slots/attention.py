# Slot: attention (v4, by Shubh Goyal)

class Attention(nn.Module):
    """Multi-head causal self-attention with Rotary Positional Embeddings (RoPE)."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_dropout = cfg.dropout

        # Precompute RoPE frequencies as a buffer (portable across devices,
        # same reasoning as pos_ids: never build device-pinned tensors in
        # forward()).
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(cfg.block_size).float()
        freqs = torch.outer(t, inv_freq)  # (block_size, head_dim/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def _apply_rope(self, x, T):
        # x: (B, n_head, T, head_dim)
        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,head_dim/2)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos
        out = torch.stack([rot_x1, rot_x2], dim=-1).flatten(-2)
        return out

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
