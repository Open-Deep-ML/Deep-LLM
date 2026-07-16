# Slot: lr_schedule (v0, by Deep-ML)

def get_lr(step, cfg):
    """Linear warmup then cosine decay to 10% of base lr."""
    warmup = 100
    if step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.max_steps - warmup)
    return cfg.learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
