# Evaluation and checkpoint selection

The three selected checkpoints cover different tradeoffs. v178/7750 is the
current video policy. v152/7650 completed all 120 archived trials, but its motion
review does not show universal jitter improvement. v171/7650 adds acceleration
penalties and is useful as a comparison for v178's action-bound penalty.
The selection does not establish a single best policy for every metric.

Each 120-case batch includes 30-second trials for stop, forward, backward, left,
right, four diagonal commands and a switching sequence. Seeds are 31000–31002.
Each backend tests both 1 ms and 0.5 ms physics with a 10 ms control interval.
The initial pose has an opposed thumb; reset perturbations use the archived
`noise=1` convention. Inference is deterministic. A physical fall or forbidden
palm support terminates a trial.

`goal_review.json` records backend-specific completion and tracking statistics.
`evaluation_*.json` retains per-trial results. Host-specific path prefixes were
removed for publication; embedded source hashes refer to the original reports
before that path normalization. Checkpoint hashes and the tensor-preservation
check are listed separately in `checkpoints/manifest.json`.

The quoted v178 cardinal velocities are signed means along the requested axis,
over completed full trials. Failed trials are excluded from that mean and are
counted separately in completion. They are not instantaneous speed magnitudes.
The target of smooth, reliable 0.1 m/s multidirectional motion remains unmet.
In particular, a high completion rate alone cannot prove low jitter.

The demo uses a different seed and smooth forward/left/backward command blends.
Its amplitude is 0.15 m/s, above the training target, and it excludes rightward
commands. No external propulsion forces are applied. Camera following and
orbiting affect presentation, so use the numeric reports to assess tracking.
The 20-second demonstration completed without a fall; this single rollout is
not a substitute for the full evaluation batch.
