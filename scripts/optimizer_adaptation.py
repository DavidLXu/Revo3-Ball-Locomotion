"""Explicit learning-rate migration, preserving every other Adam field."""
import copy
import math


def validate(config):
    assert set(config) == {'version', 'group_learning_rates', 'schedule'}
    assert config['version'] == 1 and config['schedule'] == 'fixed'
    rates = config['group_learning_rates']
    assert len(rates) == 4 and all(math.isfinite(v) and 0 < v <= .001 for v in rates)


def expected_optimizer_state(source, config):
    validate(config)
    result = copy.deepcopy(source)
    assert len(result['param_groups']) == 4
    for group, rate in zip(result['param_groups'], config['group_learning_rates']):
        group['lr'] = rate
    return result


def configure(runner, config):
    validate(config)
    previous = [g['lr'] for g in runner.alg.optimizer.param_groups]
    assert len(previous) == 4
    for group, rate in zip(runner.alg.optimizer.param_groups, config['group_learning_rates']):
        group['lr'] = rate
    runner.alg.learning_rate = config['group_learning_rates'][0]
    runner.alg.schedule = config['schedule']
    return dict(previous_learning_rates=previous,
                current_learning_rates=list(config['group_learning_rates']),
                adam_slots_and_other_group_fields_preserved=True)
