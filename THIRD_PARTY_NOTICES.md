# Third-party assets and libraries

Original code and released policy checkpoints in this repository use the MIT
license. This grant does not cover upstream robot descriptions, meshes,
textures, or external software.

The hand model comes from [BrainCoTech/brainco-description](https://github.com/BrainCoTech/brainco-description),
commit `f197d9d720b1b106a3298e2438ff07e055bf15ec`. At that revision, its README
states that license information will be provided with the public release.
This repository therefore does not redistribute its meshes, textures, or MJCF.
`scripts/setup_assets.py` downloads the pinned source from its original host
and generates a scene on the user's machine. Consult the upstream terms before
redistributing those assets. The demo is a simulation rendering of that model.

Runtime libraries retain their respective licenses: MuJoCo, MuJoCo Warp,
MjLab, NVIDIA Warp, RSL-RL, PyTorch and the packages listed in requirements.txt.

The project also explored a ground-locomotion teacher from
[DavidLXu/Revo3-Ground-Locomotion](https://github.com/DavidLXu/Revo3-Ground-Locomotion).
The three released policies load directly for inference; no external teacher
checkpoint is required. They are continued ball-task policies, not fresh
training reproductions or claims of hardware transfer.
