"""Explicit control-step history, staged independently of the active trainer.

Call reset after obtaining a reset world's first observation, and advance once
after each control interval. Reads never advance time. This stores observations
only: it does not filter observations/actions or change physical control.
"""
import torch


class TemporalObservationBuffer:
    def __init__(self, num_envs, feature_count, *, past_samples=7, stride=2,
                 device="cpu", dtype=torch.float32):
        if min(num_envs, feature_count, past_samples, stride) < 1:
            raise ValueError("Buffer dimensions and sampling stride must be positive")
        self.num_envs = num_envs
        self.feature_count = feature_count
        self.past_samples = past_samples
        self.stride = stride
        self.capacity = past_samples * stride + 1
        self.frames = torch.zeros((num_envs, self.capacity, feature_count),
                                  device=device, dtype=dtype)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.cursor = 0
        self.control_steps = 0

    def _check(self, features, rows):
        if features.shape != (rows, self.feature_count):
            raise ValueError("Feature batch shape does not match buffer")
        if features.device != self.frames.device or features.dtype != self.frames.dtype:
            raise ValueError("Feature device/dtype does not match buffer")

    @torch.no_grad()
    def reset(self, ids, features):
        """Repeat the first frame into the entire selected world's past."""
        ids = torch.as_tensor(ids, device=self.frames.device, dtype=torch.long)
        if ids.ndim != 1 or len(ids.unique()) != len(ids):
            raise ValueError("Reset ids must be a vector of unique world indices")
        if bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError("Reset index outside buffer")
        self._check(features, len(ids))
        self.frames[ids] = features[:, None, :]
        self.initialized[ids] = True

    @torch.no_grad()
    def replace_current(self, features):
        """Refresh same-time inputs (e.g. a command) without advancing history."""
        self._check(features, self.num_envs)
        if not bool(self.initialized.all()):
            raise RuntimeError("Reset all worlds before refreshing history")
        self.frames[:, self.cursor].copy_(features)

    @torch.no_grad()
    def advance(self, features):
        """Append one post-control observation for every world exactly once."""
        self._check(features, self.num_envs)
        if not bool(self.initialized.all()):
            raise RuntimeError("Reset all worlds before advancing history")
        self.cursor = (self.cursor + 1) % self.capacity
        self.frames[:, self.cursor].copy_(features)
        self.control_steps += 1

    def read(self):
        """Return independent oldest-to-newest frames, including the current one.

        Default ages are 14,12,10,8,6,4,2,0 control intervals. At 100 Hz this
        spans 140 ms. Returned tensors cannot mutate stored history.
        """
        if not bool(self.initialized.all()):
            raise RuntimeError("Reset all worlds before reading history")
        ages = torch.arange(self.past_samples, -1, -1,
                            device=self.frames.device) * self.stride
        indices = (self.cursor - ages) % self.capacity
        return self.frames.index_select(1, indices)
