# Slot: optimizer (v5, by Giuseppe Frigeni)

def configure_optimizer(model, cfg):
    """AdamW with weight decay on matrices only (vanilla nanoGPT)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.03},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )
