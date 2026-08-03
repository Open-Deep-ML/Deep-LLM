# Slot: lr_schedule (v11, by noah lin)

def get_lr(step, cfg):
    max_steps = max(1, int(cfg.max_steps))
    warmup_steps = 60
    cooldown_steps = 240
    min_lr_ratio = 0.1

    if step < warmup_steps:
        return cfg.learning_rate * (step + 1) / warmup_steps

    cooldown_start = max_steps - cooldown_steps
    if step < cooldown_start:
        return cfg.learning_rate

    progress = (step - cooldown_start) / max(1, cooldown_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.learning_rate * (
        min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    )
