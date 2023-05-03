from architecture.transformer import TransformerArchitecture
from modeling.llama import TransformerLlama


class LlamaArchitecture(TransformerArchitecture):
    def create_body(self, n_features: int, max_len: int, drop_rate: float = 0.1, depth: int = 12, n_heads: int = 8,
                    mult_factor: int = 4):
        return TransformerLlama(n_features, n_heads, max_len, drop_rate, depth, mult_factor)
