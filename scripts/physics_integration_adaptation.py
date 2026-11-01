"""Explicit integration-only migration; the policy control period is unchanged."""
CONFIG = dict(version=1, physics_timestep_s=.0005, alternate_physics_timestep_s=.0005,
              control_timestep_s=.01, decimation=20)


def validate(config):
    assert config == CONFIG, 'Unsupported physics integration adaptation'


def validate_environment(cfg, config):
    validate(config)
    assert cfg['physics_timestep'] == config['physics_timestep_s']
    assert cfg['alternate_physics_timestep'] == config['alternate_physics_timestep_s']
    assert cfg['decimation'] == config['decimation']
    assert cfg['physics_timestep'] * cfg['decimation'] == config['control_timestep_s']
    assert not cfg.get('physics_timestep_observation') and not cfg.get('split_timestep_projection')


def validate_source(cfg, saved_config, requested_config):
    if saved_config is None:
        assert cfg['physics_timestep'] == .001 and cfg['alternate_physics_timestep'] == .0005
        assert cfg['decimation'] == 10
    else:
        validate_environment(cfg, saved_config)
        assert requested_config == saved_config, 'Preserve saved integration adaptation explicitly'


def configure(cfg, config):
    validate(config)
    assert cfg['physics_timestep'] * cfg['decimation'] == config['control_timestep_s']
    before = {k:cfg[k] for k in ('physics_timestep','alternate_physics_timestep','decimation')}
    cfg.update(physics_timestep=config['physics_timestep_s'],
               alternate_physics_timestep=config['alternate_physics_timestep_s'],
               decimation=config['decimation'])
    validate_environment(cfg, config)
    return dict(before=before, after={k:cfg[k] for k in before},
                control_timestep_unchanged=True)
