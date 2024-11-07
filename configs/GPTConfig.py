import mindspore
from models.gpt.src.utils import GPTConfig

config = GPTConfig(batch_size=4,
                       seq_length=1024,
                       vocab_size=50257,
                       embedding_size=1024,
                       num_layers=24,
                       num_heads=16,
                       expand_ratio=4,
                       post_layernorm_residual=False,
                       dropout_rate=0.1,
                       compute_dtype=mindspore.float32,
                       use_past=False)
