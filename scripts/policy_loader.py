"""Select an explicitly saved representation for evaluation and diagnostics."""
import json
from pathlib import Path


def load_controller(checkpoint):
    folder = Path(checkpoint).parent
    cfg = json.loads((folder / 'runner_config.json').read_text())
    if cfg['actor']['class_name'] == 'history_context:HistoryContextModel':
        from history_lifecycle import HistoryPolicyController
        return HistoryPolicyController(checkpoint)
    from teleop_control import PolicyController
    return PolicyController(checkpoint)
