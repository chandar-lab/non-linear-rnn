from .fast_lstm import FastLSTMModel
from .gru import GRUModel
from .lstm import LSTMModel
from .nru import NRUModel

MODELS = {
    "nru": NRUModel,
    "gru": GRUModel,
    "lstm": LSTMModel,
    "fast_lstm": FastLSTMModel,
}


def build_model(cfg, vocab_size, output_size):
    """Build a model from the `model` config section; all keys except `name` go to the constructor."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', available: {list(MODELS)}")
    return MODELS[name](vocab_size, output_size, **cfg)
