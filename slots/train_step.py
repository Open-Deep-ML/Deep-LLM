# Slot: train_step (v13, by Đức Dũng Hoàng)

def train_step(model, batch, optimizer, step):
    """Mixed-precision step: fp16 forward/backward on tensor cores, fp32 master weights."""
    use_amp = next(model.parameters()).device.type == "cuda"

    if not hasattr(train_step, "scaler"):
        train_step.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler = train_step.scaler

    x, y = batch
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()
