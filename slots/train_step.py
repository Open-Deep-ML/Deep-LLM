# Slot: train_step (v0, by Deep-ML)

def train_step(model, batch, optimizer, step):
    """Plain cross-entropy step with grad clipping (vanilla nanoGPT)."""
    x, y = batch
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()
