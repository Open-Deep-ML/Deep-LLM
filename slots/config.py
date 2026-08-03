# Slot: config (v11, by noah lin)

def configure_model(cfg):
    """更小模型，换更多 tokens。"""
    cfg.n_layer = 3          # 4 → 3
    cfg.n_head = 4
    cfg.n_embd = 256         # 384 → 256（最关键的降参）
    cfg.block_size = 128
    cfg.dropout = 0.0
    cfg.batch_size = 120     # 模型小了，可以稍微加大 batch 再榨一点 tokens
    cfg.learning_rate = 1.2e-3  # 小模型通常可以稍微抬高一点 base lr
    return cfg
