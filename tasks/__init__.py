from .adding import Adding
from .associative_recall import AssociativeRecall
from .base import Task
from .copy import Copy
from .copying_memory import CopyingMemory
from .denoising import Denoising
from .psmnist import PSMNIST
from .repeat_copy import RepeatCopy

TASKS = {
    "copying_memory": CopyingMemory,
    "denoising": Denoising,
    "adding": Adding,
    "copy": Copy,
    "repeat_copy": RepeatCopy,
    "associative_recall": AssociativeRecall,
    "psmnist": PSMNIST,
}


def build_task(cfg):
    """Build a task from the `task` config section; all keys except `name` go to the constructor."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in TASKS:
        raise ValueError(f"unknown task '{name}', available: {list(TASKS)}")
    return TASKS[name](**cfg)
