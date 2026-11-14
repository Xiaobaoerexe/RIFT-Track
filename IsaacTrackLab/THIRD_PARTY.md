# Third-party pins

- Isaac Lab: `v3.0.0-EA`, commit `ae37b028ea415c91ea2bc32609efcd759ed2b974`.
- Unitree ROS: commit `15ebe53f4cd77e7d745ae83b9d7ff90e685e8998`; As2 description under `robots/as2_description`.
- Unitree SDK2: commit `d1abf001220d3e5f59e00a0c4aaa595e00089a4a`; the As2 deployment bridge links the upstream `unitree_sdk2` target and uses `unitree::robot::a2::SportClient`.
- Poly Haven: the compact HD street uses 2K PBR textures, the `urban_street_01` HDRI, and six prop models downloaded through the Poly Haven public API. Assets are CC0; exact source IDs, paths, sizes, and SHA-256 hashes are recorded in `assets/external/polyhaven/manifest.json`. Powered by Poly Haven: <https://polyhaven.com>.
- NVIDIA Isaac Sim 6.1 character assets: business, police, medical, and construction characters under `Isaac/People/Characters`. Use and redistribution remain governed by the NVIDIA/Omniverse asset terms shipped with Isaac Sim; these are referenced rather than repackaged by the HD street builder.

Before distributing any derived simulator package, review the license files in all three repositories and record the licenses/sources of every city, character, animation and prop asset in `configs/assets.json` or a generated asset manifest.
