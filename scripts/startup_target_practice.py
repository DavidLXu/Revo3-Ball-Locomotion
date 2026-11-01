"""Staged training-only command sampler intervention; no physical assistance."""
import types
import torch


def install_startup_target_practice(env, speed):
    assert 0 < speed <= env.cfg['target_speed']
    assert env.cfg['reversal_sequence_probability'] == 0
    for group in env.groups:
        assert group.startup_env_count > 0
        assert not hasattr(group, 'startup_practice_original_sampler')
        original = group.sample_commands
        group.startup_practice_original_sampler = original

        def sample(this, ids, original=original):
            original(ids)
            selected = ids[ids < this.startup_env_count]
            targets = this.command_targets[selected]
            magnitude = targets.norm(dim=-1)
            moving = selected[magnitude > 1e-6]
            if len(moving):
                this.command_targets[moving] *= speed / magnitude[magnitude > 1e-6, None]

        group.sample_commands = types.MethodType(sample, group)


def check_sampler_invariants(env, speed):
    """Matched RNG checks on full, partial, nonstartup and empty ID sets."""
    checks = 0
    for group in env.groups:
        n = group.num_envs
        subsets = [torch.arange(n, device=env.device),
                   torch.arange(group.startup_env_count, n, device=env.device),
                   torch.empty(0, device=env.device, dtype=torch.long)]
        subsets += [torch.randperm(n, device=env.device)[:n//3] for _ in range(10)]
        for ids in subsets:
            rng = torch.cuda.get_rng_state(env.device)
            group.startup_practice_original_sampler(ids)
            target = group.command_targets.clone()
            countdown = group.command_countdown.clone()
            after_rng = torch.cuda.get_rng_state(env.device)
            torch.cuda.set_rng_state(rng, env.device)
            group.sample_commands(ids)
            assert torch.equal(torch.cuda.get_rng_state(env.device), after_rng)
            assert torch.equal(countdown, group.command_countdown)
            expected = target.clone()
            selected = ids[ids < group.startup_env_count]
            magnitude = target[selected].norm(dim=-1)
            moving = selected[magnitude > 1e-6]
            expected[moving] *= speed / magnitude[magnitude > 1e-6, None]
            assert torch.equal(expected, group.command_targets)
            assert torch.equal(target.norm(dim=-1) < 1e-6,
                               group.command_targets.norm(dim=-1) < 1e-6)
            checks += 1
    return checks
