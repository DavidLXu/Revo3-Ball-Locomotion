"""Explicit, opt-in hand inertia variant; legacy task models remain unchanged."""
from pathlib import Path

import mujoco
import numpy as np


def collision_only_hand_inertia(model, initial_pose='opposed_thumb'):
    """Exclude hand visual meshes from automatic inertia inference only.

    Preserve explicit vendor inertials, all ball properties, contact geometry,
    approved pose, joints, controller and solver. Recompute derived constants.
    This aligns inference with vendor URDF import, not hardware calibration.
    """
    root = Path(__file__).resolve().parents[1]
    assert initial_pose == 'opposed_thumb'
    spec = mujoco.MjSpec.from_file(str(root/'outputs/opposed_thumb/scene.xml'))
    changed_geoms = []
    for geom in spec.geoms:
        if geom.name.startswith('hand/') and geom.group == 2:
            geom.mass = 0.
            changed_geoms.append(geom.name)
    assert changed_geoms
    reference = spec.compile()
    assert reference.nbody == model.nbody
    for field in ('body_parentid', 'body_pos', 'body_quat', 'geom_type',
                  'geom_bodyid', 'geom_pos', 'geom_quat', 'geom_size',
                  'geom_contype', 'geom_conaffinity', 'geom_friction',
                  'jnt_type', 'jnt_range', 'jnt_qposadr', 'jnt_dofadr'):
        assert np.array_equal(getattr(reference, field), getattr(model, field)), field
    fields = ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat')
    hand_ids = [i for i in range(model.nbody) if model.body(i).name.startswith('hand/')]
    before = float(model.body_mass[hand_ids].sum())
    ball = model.body('ball/ball').id
    ball_before = {key: getattr(model, key)[ball].copy() for key in fields}
    for i in hand_ids:
        assert reference.body(i).name == model.body(i).name
        for field in fields:
            getattr(model, field)[i] = getattr(reference, field)[i]
    mujoco.mj_setConst(model, mujoco.MjData(model))
    for field, value in ball_before.items():
        assert np.array_equal(getattr(model, field)[ball], value)
    after = float(model.body_mass[hand_ids].sum())
    assert abs(before-2.3057722099783238) < 1e-6
    assert abs(after-1.7640477535610517) < 1e-6
    return {'mode': 'collision_only', 'hand_mass_before_kg': before,
            'hand_mass_after_kg': after, 'ball_mass_kg': float(model.body_mass[ball]),
            'excluded_visual_geoms': len(changed_geoms), 'ball_properties_unchanged': True}
