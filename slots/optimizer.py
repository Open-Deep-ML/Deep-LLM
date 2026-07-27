# Slot: optimizer (v8, by Nick Grebe)

def configure_optimizer(model, cfg):
    """AdamW with weight decay applied only to matrix-shaped parameters."""

    decay = [p for p in model.parameters()
        if p.requires_grad and p.dim() >= 2
    ]

    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    optimizer_args = {
        "lr": cfg.learning_rate,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }

    if next(model.parameters()).device.type == "cuda":
        optimizer_args["fused"] = True

    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": 0.05,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        **optimizer_args,
    )
