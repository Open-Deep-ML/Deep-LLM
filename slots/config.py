# Slot: config (v3, by Giuseppe Frigeni)

def configure_model(cfg):
    """The model's shape and training hyperparameters (vanilla nanoGPT).

    Everything here is a tradeoff against the fixed compute budget: deeper
    or wider means fewer steps before the wall clock; longer context means
    slower steps but more to attend to. The param cap is the hard ceiling.
    """
    cfg.n_layer = 6
    cfg.n_head = 6
    cfg.n_embd = 384
    cfg.block_size = 128      # context length (max 1024)
    cfg.dropout = 0.1
    cfg.batch_size = 64
    cfg.learning_rate = 3e-4
    return cfg
