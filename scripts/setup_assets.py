"""Fetch pinned upstream assets and recreate the checkpoint's exact reset pose."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'f197d9d720b1b106a3298e2438ff07e055bf15ec'
URL = 'https://github.com/BrainCoTech/brainco-description.git'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-download', action='store_true', help='Use existing assets and verify their pinned per-file checksums.')
    args = parser.parse_args()
    assets = ROOT / 'assets/brainco-description'
    if not assets.exists():
        if args.skip_download:
            parser.error('Pinned upstream checkout is missing')
        assets.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', URL, str(assets)], check=True)
        subprocess.run(['git', '-C', str(assets), 'sparse-checkout', 'set', '--no-cone', '/revo3_system/mjcf/revo3_right.xml', '/revo3_system/meshes/hands/visual/right/', '/revo3_system/meshes/hands/collision/right/', '/revo3_system/meshes/hands/visual/textures/'], check=True)
        subprocess.run(['git', '-C', str(assets), 'checkout', REVISION], check=True)
    manifest = json.loads((ROOT / 'configs/upstream_assets.json').read_text())
    for entry in manifest['files']:
        path = assets / entry['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError(f'Missing or changed upstream asset: {entry["path"]}. Expected revision {REVISION}.')
    import numpy as np
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from build_scene import hand_spec, ball_spec, decorate, SOURCE, RADIUS
    cfg = SceneCfg(num_envs=1, env_spacing=1, entities={
        'hand': EntityCfg(spec_fn=hand_spec),
        'ball': EntityCfg(spec_fn=ball_spec, init_state=EntityCfg.InitialStateCfg(pos=(0, 0, RADIUS)))}, spec_fn=decorate)
    scene = Scene(cfg, device='cpu')
    directory = ROOT / 'outputs/opposed_thumb'
    directory.mkdir(parents=True, exist_ok=True)
    xml = ET.fromstring(scene.spec.to_xml())
    for asset in xml.find('asset'):
        if asset.get('file'):
            source = Path(asset.get('file'))
            if not source.is_absolute():
                source = (SOURCE.parent / source).resolve()
            asset.set('file', os.path.relpath(source, directory))
    old = xml.find('keyframe')
    if old is not None:
        xml.remove(old)
    qpos = np.load(ROOT / 'configs/pose/initial_qpos.npy')
    ET.SubElement(ET.SubElement(xml, 'keyframe'), 'key', name='opposed_thumb_initial', qpos=' '.join(map(str, qpos)))
    tree = ET.ElementTree(xml)
    ET.indent(tree)
    tree.write(directory / 'scene.xml', encoding='unicode')
    for name in ('initial_qpos.npy', 'initial_state.json'):
        shutil.copy2(ROOT / 'configs/pose' / name, directory / name)
    for checkpoint_dir in (ROOT / 'checkpoints').iterdir():
        if checkpoint_dir.is_dir():
            for name in ('initial_qpos.npy', 'initial_state.json', 'scene.xml'):
                shutil.copy2(directory / name, checkpoint_dir / ('pose_' + name))
    (ROOT / 'runs').mkdir(exist_ok=True)
    from balance_model import make_model
    model, _, _, _ = make_model('opposed_thumb', 'collision_only')
    assert model.nu == 21
    print(f'Prepared opposed-thumb pose: nq={model.nq}, nv={model.nv}, actuators={model.nu}')


if __name__ == '__main__':
    main()
