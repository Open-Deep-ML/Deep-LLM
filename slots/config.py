# Slot: config (v16, by Đức Dũng Hoàng)

def configure_model(cfg):
    cfg.n_layer = 4
    cfg.n_head = 4
    cfg.n_embd = 512
    cfg.block_size = 512
    cfg.dropout = 0.15
    cfg.batch_size = 32
    cfg.learning_rate = 1.5e-3
    return cfg
