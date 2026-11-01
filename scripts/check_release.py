"""Verify published checkpoint hashes, then run a two-second native smoke test."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from policy_loader import load_controller

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    entries = json.loads((ROOT / 'checkpoints/manifest.json').read_text())['checkpoints']
    for entry in entries:
        path = ROOT / entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path
        controller = load_controller(path)
        env = controller.env
        env.reset(31000, 1.)
        for _ in range(200):
            env.set_command(.1, 0.)
            action = controller.action()
            assert np.isfinite(action).all()
            env.step(action)
            assert not env.metrics()['failed'], path
        print(f"PASS {entry['path']}: hash verified, 2 s forward rollout", flush=True)


if __name__ == '__main__':
    main()
