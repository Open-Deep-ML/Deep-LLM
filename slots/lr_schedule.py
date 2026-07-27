# Slot: lr_schedule (v8, by Nick Grebe)

def get_lr(step, cfg):
    """3% linear warmup, then cosine decay to 10% of the base learning rate."""
    max_steps = max(1,int(cfg.max_steps),)

    warmup_steps = max(1,
        min(200,
            int(0.03 * max_steps),
        ),
    )

    if step < warmup_steps:
        return (cfg.learning_rate* (step + 1)/ warmup_steps
        )

    progress = (step - warmup_steps) / max(1,max_steps - warmup_steps - 1,)

    progress = min(1.0,max(0.0, progress),)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return cfg.learning_rate * (0.1 + 0.9 * cosine)
