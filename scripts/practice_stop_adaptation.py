"""Explicit training command-duration adaptation; no inference or physics change."""

CONFIG = dict(version=1, practice_stop_hold_max_s=5.)


def validate_environment(cfg, config):
    assert config == CONFIG, 'Unsupported practice stop adaptation'
    assert cfg['practice_stop_hold_max'] == config['practice_stop_hold_max_s']
    assert cfg['target_speed'] == .1 and cfg['practice_speeds'] == [.1]


def validate_source(cfg, saved_config, requested_config):
    if saved_config is not None:
        validate_environment(cfg, saved_config)
        assert requested_config == saved_config, 'Preserve saved command adaptation explicitly'
    elif requested_config is not None:
        assert requested_config == CONFIG
        assert cfg['practice_stop_hold_max'] == 1., 'Expected the short-stop comparison source'


def configure(cfg, config):
    assert config == CONFIG
    before = cfg['practice_stop_hold_max']
    cfg['practice_stop_hold_max'] = config['practice_stop_hold_max_s']
    validate_environment(cfg, config)
    return dict(before_s=before, after_s=cfg['practice_stop_hold_max'])
