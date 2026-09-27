class Task:
    """Interface for sequence tasks. Everything is tokens in, classes out, one prediction per input position.

    Batches are (x, y, mask), all of shape (batch, length):
        x     input token ids in [0, input_size)
        y     target class ids in [0, output_size)
        mask  bool; only True positions count towards the loss (cross-entropy) and metrics

    Generator tasks implement sample() and work with both data modes (stream / file).
    Dataset tasks (fixed data such as MNIST) set is_dataset = True and implement load_splits() and make_batch().
    """
    input_size: int
    output_size: int
    is_dataset = False

    def sample(self, batch_size, generator):
        """Return (x, y, mask) for a fresh batch. All examples in a batch share a length."""
        raise NotImplementedError

    def load_splits(self, data_dir, num_val, seed):
        """Dataset tasks: return {"train": TensorDataset, "val": TensorDataset, optionally "test": ...}."""
        raise NotImplementedError

    def make_batch(self, *tensors):
        """Dataset tasks: turn a slice of a split's TensorDataset into (x, y, mask)."""
        raise NotImplementedError

    def metrics(self, logits, y, mask):
        """Scalar metrics for a batch: accuracy over masked positions, and fraction of fully correct sequences."""
        correct = (logits.argmax(-1) == y) | ~mask
        return {"acc": correct[mask].float().mean().item(), "seq_acc": correct.all(-1).float().mean().item()}

    def format_sample(self, x, y, mask, logits, max_items=20):
        """Rich markup string comparing target and prediction at the masked positions of one example."""
        target, pred = y[mask][:max_items].tolist(), logits[mask][:max_items].argmax(-1).tolist()
        shown = " ".join(f"[green]{p}[/]" if p == t else f"[red]{p}[/]" for p, t in zip(pred, target))
        return f"target {' '.join(map(str, target))}  │  pred {shown}"

    @property
    def baseline_loss(self):
        """Loss of a trivial reference solution, or None."""
        return None
