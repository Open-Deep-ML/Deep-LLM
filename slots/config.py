# Slot: config (v15, by Nguyễn Đức Bảo Lâm)

def configure_model(cfg):
    cfg.n_layer = 4
    cfg.n_head = 4
    cfg.n_embd = 512
    cfg.block_size = 512
    cfg.dropout = 0.25
    cfg.batch_size = 32
    cfg.learning_rate = 1.5e-3
    return cfg
