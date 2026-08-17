# Slot: config (v12, by Đức Dũng Hoàng)

def configure_model(cfg):
    cfg.n_layer = 3
    cfg.n_head = 4
    cfg.n_embd = 256
    cfg.block_size = 256
    cfg.dropout = 0.0
    cfg.batch_size = 60
    cfg.learning_rate = 1.2e-3
    return cfg
