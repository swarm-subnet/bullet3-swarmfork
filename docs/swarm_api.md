# Swarm additions to PyBullet

Everything this fork adds on top of upstream Bullet and PyBullet, in one place: every render flag, camera argument, visual shape flag, collision flag, environment variable and build switch, with its value, what it does, what it costs and the pull request that added it. Values are read from the source on `master`; where a pull request text carries an older number, the source wins.

The distribution is `swarm-bullet3` and the module is still `import pybullet`. A wheel carries a switch when the constant exists: `hasattr(pybullet, "ER_SWARM_RAYCAST")` is the check the swarm repository uses before it asks for one.

## Where the shipped code lives

| Part | Path |
|---|---|
| Python module | `src/pybullet/pybullet.c` (constants exported near the end of `initpybullet`) |
| Physics server, commands, public enums | `src/SharedMemory/`, flags in `SharedMemoryPublic.h` |
| Software renderer (TinyRenderer) | `src/TinyRenderer/` |
| Renderer plugin: the converter, the ray caster, the sky | `src/SharedMemory/plugins/tinyRendererPlugin/` (`TinyRendererVisualShapeConverter.cpp`, `SwarmRaycast.cpp`, `SwarmSky.cpp`) |
| Mesh and texture import | `examples/Importers/` |
| Embree build recipe and patches | `examples/ThirdPartyLibs/embree/` |
| Python tests of every switch | `examples/pybullet/unittests/` |
| Build | `setup.py`, `wheel.sh` |

`examples/` is upstream's name for the folder; the helper libraries the server includes still live there. The three parts that make the wheel were moved into `src/` in [#20](https://github.com/swarm-subnet/bullet3-swarmfork/pull/20).

## Rules every change keeps

- Validators are CPU only. Nothing here needs a GPU.
- The bytes of an image, and so a score, must be the same on every validator: one binary, one code path, no runtime CPU dispatch, no fast-math, `-ffp-contract=off`, hardware reciprocals in Embree replaced by IEEE division, tree builds on one thread, pixels dealt to threads in fixed 16 x 16 tiles (tile k goes to thread k mod T), no maths-library transcendental in a per-pixel path (the sky and the gamma table use polynomials and literal tables).
- Every switch is off by default and every existing challenge family renders the same bytes as before. The proof is `validator/scripts/verify_render_identity.py` in the swarm repository (7 scenes, orbit and episode hashes) run on the wheel before and after a change.
- A new render flag takes the next free power of two; the next free value is 16384.

## 1. Render flags

Passed in `flags=` of `getCameraImage` and `getDepthImagesBatch`, combined with `|`. The first three are upstream. "Rasteriser" is TinyRenderer's triangle path; "ray cast" is the Embree path behind `ER_SWARM_RAYCAST`.

| Flag | Value | Path | What it does | Cost, measured | Added |
|---|---|---|---|---|---|
| `ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX` | 1 | both | Mask pixel = `objectId + ((linkIndex + 1) << 24)` | upstream | upstream |
| `ER_USE_PROJECTIVE_TEXTURE` | 2 | rasteriser | upstream projective texture | upstream | upstream |
| `ER_NO_SEGMENTATION_MASK` | 4 | both | No mask buffer | upstream | upstream |
| `ER_DEPTH_ONLY` | 8 | both | Skips colour shading, sky, glint and shadow; the rasteriser culls bodies outside the frustum and triangles past the far plane and runs its face loop on OpenMP threads | the depth families' default; a 128 px mountain depth frame is 16 ms rasterised, 3.4 ms ray cast at 2 threads | before the programme ([#1](https://github.com/swarm-subnet/bullet3-swarmfork/pull/1), [#2](https://github.com/swarm-subnet/bullet3-swarmfork/pull/2)) |
| `ER_TEXTURE_FILTER` | 16 | both, colour | Bilinear texture reads from a mip chain built once per texture on the first filtered frame; the level comes from the pixel footprint, from float exponent bits, not `log2` | office 256 px, 1 thread: +2 ms per frame, +100 ms first frame, +41 MB for the office textures | [#9](https://github.com/swarm-subnet/bullet3-swarmfork/pull/9) |
| `ER_SWARM_RAYCAST` | 32 | ray cast | Depth, mask and colour through Embree: the static bodies in one world-space tree built once per world, bodies that move as instances of shared per-mesh trees, one ray per pixel corner, TinyRenderer's clip-z convention, colour shaded in C++ at the hit with TinyRenderer's light formula; `shadow=1` casts one occlusion ray per hit | cost follows pixels, not triangles: 256 px colour at 2 threads is 20 ms on the office (rasteriser 41 ms), 20 ms on the solar-park slice (rasteriser 112 ms), 27 ms on a 1M-triangle grid (rasteriser 207 ms); first frame pays the tree build, 0.3 to 1.9 s per map | [#11](https://github.com/swarm-subnet/bullet3-swarmfork/pull/11), [#12](https://github.com/swarm-subnet/bullet3-swarmfork/pull/12), [#13](https://github.com/swarm-subnet/bullet3-swarmfork/pull/13) |
| `ER_SWARM_SHADOW_MAP` | 64 | ray cast, needs `shadow=1` | The light's view of the static bodies is cast once into a depth grid (about 4096 cells a side plus one margin cell on each edge, never finer than 5 mm) and each hit looks itself up instead of firing a shadow ray; a new light direction casts the grid again; a body that moves or hides recasts only its own cells | shadow cost 6 to 12 ms per frame drops to under 1 ms; grid build 0.5 s office, 1.5 s solar park, once per world and light | [#14](https://github.com/swarm-subnet/bullet3-swarmfork/pull/14) |
| `ER_SWARM_MOVER_SHADOW` | 128 | ray cast, needs `ER_SWARM_SHADOW_MAP` | Bodies that moved since the world was built sit in a second small tree; a hit the map calls lit fires one short ray at it, so movers cast shadows | 1 to 3 ms per frame, independent of the number of movers | [#15](https://github.com/swarm-subnet/bullet3-swarmfork/pull/15) |
| `ER_EDGE_ANTIALIAS` | 256 | ray cast, colour | A pixel is an edge when a neighbour hit another body or its 1/z is off the line through its neighbours by more than 0.1 %; edge pixels are recomposited from the exact share of the pixel each nearby triangle covers, one probe ray for the uncovered rest; depth and mask keep the first ray | +3 to 5 ms on the real maps (6 to 10 % of pixels are edges) | [#16](https://github.com/swarm-subnet/bullet3-swarmfork/pull/16) |
| `ER_ALPHA_CUTOUT` | 512 | ray cast | A hit on a texel whose alpha is below 128 is a miss for camera and shadow rays alike, so colour, depth, mask and shadow share the same holes; textures keep their alpha plane at decode for this | about 1 ms on a 256 px colour frame, 0.5 ms on a 128 px depth frame | [#17](https://github.com/swarm-subnet/bullet3-swarmfork/pull/17) |
| `ER_SPECULAR_GLINT` | 1024 | both, colour | A hit blends towards the sky colour of its mirrored view direction, weighted by the object's specular colour times a glass Fresnel curve (`0.04 + 0.96 (1 - cos)^5`); `createVisualShape` sends specular white when none is given, so matte pieces need `specularColor=[0, 0, 0]` | under 2 ms | [#18](https://github.com/swarm-subnet/bullet3-swarmfork/pull/18) |
| `ER_SWARM_SKY_SUN` | 2048 | both, colour | A daylight sky (Preetham model, turbidity 2.5) computed from `lightDirection` and `lightColor` into six 256 x 256 cube faces, rebuilt only when the light, the cloud seed or the up axis change; a 1 degree sun disc; the sky's average tints the ambient term; with the flag on, the computed sky is painted whether or not `skyHorizonColor` and `skyZenithColor` are given; `skyCloudSeed` adds clouds | about 2 ms per frame (one cube lookup per empty pixel); map build 56 to 61 ms clear, 79 to 86 ms with clouds | [#22](https://github.com/swarm-subnet/bullet3-swarmfork/pull/22) |
| `ER_SWARM_LINEAR_LIGHT` | 4096 | ray cast, colour | The texel-times-colour byte is decoded to linear light through a 256-entry table, lit, blended and glinted there, and encoded back to sRGB on the write, so half light is byte 188 rather than 127 | +0.2 to 0.35 ms per frame | [#24](https://github.com/swarm-subnet/bullet3-swarmfork/pull/24) |
| `ER_SWARM_DAYLIGHT` | 8192 | ray cast, colour, needs `ER_SWARM_RAYCAST` | The daylight model: a hit is lit by the sky in the direction it faces (a nine-coefficient table built with the sky) plus the sun, so `lightAmbientCoeff` scales the sky and `lightDiffuseCoeff` may be several times it; the glint reflects the sky cube itself through a coated-glass curve (flat 5.5 % until the view grazes, a full mirror below cos 0.2); the shadow map is read from nine cells weighted by where the point falls in its cell (a soft edge that slides smoothly, no staircase along an edge near a grid axis) and, inside `shadowCoreRadius`, from a second 4096-cell grid about the origin; filtered textures take up to four reads along the long side of the pixel footprint; hits fade to the horizon colour with `hazeDistance`; a double-sided cut-out face seen from behind keeps its front light plus 35 % of the sun through it; the linear result times `exposure` goes through the AgX film curve (public polynomial form) to the byte. The sky is `ER_SWARM_SKY_SUN`'s Preetham sky built at 512 px per face with a bright sun disc, or the photograph of `skyTextureId`, which needs no sky flag; with neither the light is flat white | solar-park slice, 256 px, 4 threads: 18 ms to 25 ms; 960 x 540: about +40 % | [#27](https://github.com/swarm-subnet/bullet3-swarmfork/pull/27) |

Dependencies between flags: `ER_SWARM_DAYLIGHT` acts only with `ER_SWARM_RAYCAST` and takes its sky from `ER_SWARM_SKY_SUN` or from `skyTextureId`; `ER_SWARM_SHADOW_MAP` acts only with `shadow=1` and `ER_SWARM_RAYCAST`; `ER_SWARM_MOVER_SHADOW` only with `ER_SWARM_SHADOW_MAP`; `ER_EDGE_ANTIALIAS`, `ER_ALPHA_CUTOUT` and `ER_SWARM_LINEAR_LIGHT` only with `ER_SWARM_RAYCAST`; `ER_TEXTURE_FILTER`, `ER_SPECULAR_GLINT` and `ER_SWARM_SKY_SUN` work on both colour paths. `ER_DEPTH_ONLY` switches all colour work off whatever else is set.

The full picture on the ray-cast path, as measured on the solar-park slice at 256 px and 2 threads: `ER_SWARM_RAYCAST | ER_SWARM_SHADOW_MAP | ER_SWARM_MOVER_SHADOW | ER_EDGE_ANTIALIAS | ER_ALPHA_CUTOUT | ER_TEXTURE_FILTER | ER_SPECULAR_GLINT | ER_SWARM_SKY_SUN | ER_SWARM_LINEAR_LIGHT` with `shadow=1` is about 29 to 34 ms per frame, against 100 to 112 ms for the plain rasterised frame.

## 2. Camera call arguments

`getCameraImage(width, height, viewMatrix, projectionMatrix, lightDirection, lightColor, lightDistance, shadow, lightAmbientCoeff, lightDiffuseCoeff, lightSpecularCoeff, renderer, flags, projectiveTextureView, projectiveTextureProj, physicsClientId, skyHorizonColor, skyZenithColor, skyCloudSeed, shadowLightCoeff, exposure, hazeDistance, skyTextureId, skyYaw, shadowCoreRadius)`. The last nine are the fork's. They sit at the end so positional callers are unaffected.

| Argument | Type, default | What it does | Sticky? | Added |
|---|---|---|---|---|
| `skyHorizonColor`, `skyZenithColor` | RGB in 0..1, none | Where nothing is drawn, a sky blended per pixel by how far above the horizon the view ray points; one of the two alone is a flat sky; neither means the white clear as upstream; depth and mask are never touched | per call | [#10](https://github.com/swarm-subnet/bullet3-swarmfork/pull/10) |
| `skyCloudSeed` | int, none | Seeded value-noise cloud layer in the `ER_SWARM_SKY_SUN` sky, lit in the sun colour, faded at the horizon; the same seed gives the same clouds | per call | [#22](https://github.com/swarm-subnet/bullet3-swarmfork/pull/22) |
| `shadowLightCoeff` | float, 0.8 | Share of the direct light a shadowed hit keeps on the ray-cast path, from the shadow ray, the shadow map and the mover tree alike; 0.0 is a full shadow lit by the ambient term alone; the rasteriser keeps its own fixed 0.8 | sticky per client, like `lightAmbientCoeff` | [#23](https://github.com/swarm-subnet/bullet3-swarmfork/pull/23) |
| `exposure` | float, 1.0 | `ER_SWARM_DAYLIGHT`: scale on the linear light before the film curve; the sky's mean radiance is 1, so 0.45 puts a clear sky near display 0.65 | sticky | [#27](https://github.com/swarm-subnet/bullet3-swarmfork/pull/27) |
| `hazeDistance` | float, 0 | `ER_SWARM_DAYLIGHT`: metres at which a hit is 63 % the sky's colour just above the horizon in its direction; 0 is no haze | sticky | [#27](https://github.com/swarm-subnet/bullet3-swarmfork/pull/27) |
| `skyTextureId`, `skyYaw` | int from `loadTexture`, float degrees | `ER_SWARM_DAYLIGHT`: an equirectangular RGB photograph (top row the zenith, column 0 the +x heading) becomes the sky, turned by `skyYaw` about the up axis; scaled to a mean luminance of 1 under a sun 30 degrees up or higher, down to 0.2 with the sun at the horizon; texels clipped to white within 5 degrees of the light direction counted eight times brighter as the sun; absent means the Preetham sky | per call | [#27](https://github.com/swarm-subnet/bullet3-swarmfork/pull/27) |
| `shadowCoreRadius` | float, 0 | `ER_SWARM_DAYLIGHT` with `ER_SWARM_SHADOW_MAP`: half side of a second, finer shadow grid over the square about the world origin, read first for the points it covers; 0 keeps the one grid | sticky | [#27](https://github.com/swarm-subnet/bullet3-swarmfork/pull/27) |
| `shadow` | int, 0 | Upstream argument, different meaning per path: the rasteriser's shadow pass aims at the world origin from `lightDistance` (default 2 m) and does nothing useful on a map; on the ray-cast path `shadow=1` gives real shadows, from a ray or, with the flag, the map | sticky | upstream |
| `lightDirection`, `lightColor`, `lightAmbientCoeff`, `lightDiffuseCoeff`, `lightSpecularCoeff` | upstream | The light model both paths share; the swarm daylight module fills them from the seed | sticky | upstream |

`getDepthImagesBatch(width, height, viewMatrices, projectionMatrix, lightDirection, flags, physicsClientId)` is a fork-only call: several depth cameras of the same scene in one request, returned as a `(numCameras, height, width)` float32 array. Depth only, needs NumPy, accepts `ER_SWARM_RAYCAST`; all cameras of one call share one tile schedule across the threads. The swarm environment uses it for every multi-drone family unless `SWARM_BATCH_DEPTH=0`.

## 3. Visual shape flags

Passed in `flags=` of `createVisualShape` and `changeVisualShape`; `loadURDF` has its own flag for the material case. The whole enum, upstream rows included, so the taken bits are visible:

| Flag | Value | Call | What it does | Cost, measured | Added |
|---|---|---|---|---|---|
| `VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS` | 1 | `getVisualShapeData` | Upstream: report texture ids | none | upstream |
| `VISUAL_SHAPE_DOUBLE_SIDED` | 4 | create, change | Upstream: soft bodies only. Multibodies still ignore it on purpose, because our maps already pass it and repairing it would move their pixels | none | upstream |
| `VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY` | 8 | create, change | A thin shape is drawn from both sides in colour, depth and mask, on both render paths; the back side gets ambient light only | none measurable | [#4](https://github.com/swarm-subnet/bullet3-swarmfork/pull/4) |
| `VISUAL_SHAPE_MATERIALS_FROM_MTL` | 16 | create | An OBJ with several `usemtl` groups becomes one render object per material inside the same body and link, each with its `Kd` colour and `map_Kd` texture; alpha from `d` only together with `URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL`; `changeVisualShape(shapeIndex=-1)` then overrides every group | load about 30 % slower for the file, frame time unchanged | [#6](https://github.com/swarm-subnet/bullet3-swarmfork/pull/6) |
| `URDF_USE_MATERIALS_FROM_MTL` | 1 << 24 | `loadURDF` | Sets the flag above on every visual of the loaded URDF | same | [#6](https://github.com/swarm-subnet/bullet3-swarmfork/pull/6) |
| `VISUAL_SHAPE_RENDER_TREE_CACHE` | 32 | create | A static body gets its own world-space ray-cast tree, saved as `<SWARM_BVH_CACHE_DIR>/<key>.rtree` (key: mesh content hash, body transform, scale) and loaded on the next process; needs the folder to be set; a body that moves later drops the tree and carries on as a mover | first ray-cast frame on the solar park 489 ms unflagged, 813 ms cold write, 222 ms warm load; 80 MB of files for its seven pieces | [#19](https://github.com/swarm-subnet/bullet3-swarmfork/pull/19) |
| `VISUAL_SHAPE_GLASS` | 64 | create, change | A thin pane on the ray-cast path under `ER_SWARM_DAYLIGHT`: the pixel is the sky mirrored in the pane by the Fresnel of its two faces (plain glass, about 8 % head-on, all mirror when the view grazes) plus, for the rest, what the same ray meets behind the pane shaded as any hit and tinted by the pane's colour and texture; the ray passes up to three panes, so a cab is seen through both its windows; depth, mask and shadows keep the first pane as a surface; without the daylight model the bit does nothing | one more ray and one more shade on the pixels that land on glass; nothing elsewhere | [#28](https://github.com/swarm-subnet/bullet3-swarmfork/pull/28) |

The `loadURDF` flag enum in full. Only the last row is the fork's; the rest is upstream and listed so a new bit is never taken twice:

| Flag | Value | Origin |
|---|---|---|
| `URDF_USE_INERTIA_FROM_FILE` | 2 | upstream |
| `URDF_USE_SELF_COLLISION` | 8 | upstream |
| `URDF_USE_SELF_COLLISION_EXCLUDE_PARENT` | 16 | upstream |
| `URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS` | 32 | upstream |
| `URDF_RESERVED` | 64 | upstream, not exported to Python |
| `URDF_USE_IMPLICIT_CYLINDER` | 128 | upstream |
| `URDF_GLOBAL_VELOCITIES_MB` | 256 | upstream |
| `MJCF_COLORS_FROM_FILE` | 512 | upstream |
| `URDF_ENABLE_CACHED_GRAPHICS_SHAPES` | 1024 | upstream |
| `URDF_ENABLE_SLEEPING` | 2048 | upstream |
| `URDF_INITIALIZE_SAT_FEATURES` | 4096 | upstream |
| `URDF_USE_SELF_COLLISION_INCLUDE_PARENT` | 8192 | upstream |
| `URDF_PARSE_SENSORS` | 16384 | upstream, not exported to Python |
| `URDF_USE_MATERIAL_COLORS_FROM_MTL` | 32768 | upstream |
| `URDF_USE_MATERIAL_TRANSPARANCY_FROM_MTL` | 65536 | upstream |
| `URDF_MAINTAIN_LINK_ORDER` | 131072 | upstream |
| `URDF_ENABLE_WAKEUP` | 1 << 18 | upstream |
| `URDF_MERGE_FIXED_LINKS` | 1 << 19 | upstream |
| `URDF_IGNORE_VISUAL_SHAPES` | 1 << 20 | upstream |
| `URDF_IGNORE_COLLISION_SHAPES` | 1 << 21 | upstream |
| `URDF_PRINT_URDF_INFO` | 1 << 22 | upstream |
| `URDF_GOOGLEY_UNDEFINED_COLORS` | 1 << 23 | upstream |
| `URDF_USE_MATERIALS_FROM_MTL` | 1 << 24 | fork, [#6](https://github.com/swarm-subnet/bullet3-swarmfork/pull/6) |

Two related behaviours without a flag:

- `resetMeshData(bodyUniqueId, vertices, linkIndex=-1)` rewrites the vertex positions of a body's **visual** mesh in place, keeping its faces, uvs, texture and colour, rebuilding its normals from the new shape and refitting its ray-cast tree; upstream it only served deformable bodies, and a body with no soft body used to fail. The vertex count must match the mesh or the call raises, and a body whose mesh is shared with another gets its own copy on the first write. This is how an animated actor stays one surface: one body per animal, its baked frame uploaded before each picture, rather than one rigid body per bone ([#30](https://github.com/swarm-subnet/bullet3-swarmfork/pull/30)).
- `specularColor` on `createVisualShape` and `changeVisualShape`, and `Ks` in an MTL, now reach the software renderer; they are read only by `ER_SPECULAR_GLINT`. Without a value `createVisualShape` sends white ([#18](https://github.com/swarm-subnet/bullet3-swarmfork/pull/18)).
- A PNG or TGA texture with an alpha channel keeps its alpha plane at decode, in `loadTexture` and in `map_Kd`; a texture without alpha, or with alpha 255 everywhere, stores none. Only `ER_ALPHA_CUTOUT` reads it ([#17](https://github.com/swarm-subnet/bullet3-swarmfork/pull/17)).

## 4. Collision shape flags

| Flag | Value | Call | What it does | Cost, measured | Added |
|---|---|---|---|---|---|
| `GEOM_CONCAVE_BVH_CACHE` | 4 | `createCollisionShape`, with `GEOM_FORCE_CONCAVE_TRIMESH` | The concave mesh's bounding volume tree is read from `<SWARM_BVH_CACHE_DIR>/<key>.bvh` when present and written there after the first build; the key hashes every scaled triangle, the margin, the triangle count and the tree's byte layout, so a different scale or wheel never reads the wrong file; written through a temporary file and a rename; a damaged file is rebuilt over | mountain seed: collision shapes 2.7 s to 0.9 s, world build 5.2 s to 3.4 s; office 0.25 s to 0.09 s; six mountain and office seeds leave 334 files, 137 MB | [#7](https://github.com/swarm-subnet/bullet3-swarmfork/pull/7) |

`GEOM_FORCE_CONCAVE_TRIMESH` (1), `GEOM_CONCAVE_INTERNAL_EDGE` (2) and `GEOM_INITIALIZE_SAT_FEATURES` (the `URDF_INITIALIZE_SAT_FEATURES` value, 4096) in the same enum are upstream. Without the flag, or without the folder, every shape builds as upstream does.

## 5. Environment variables the engine reads at run time

| Variable | Default | Read by | What it does |
|---|---|---|---|
| `SWARM_RENDER_THREADS` | 2 | `b3GetSwarmRenderThreads` in `TinyRenderer.cpp`, once per process | OpenMP threads for the depth face loop and the ray-cast tiles, clamped to 1..16. The bytes do not depend on it: the rasteriser's depth loop writes each pixel once and the ray caster deals fixed tiles |
| `SWARM_BVH_CACHE_DIR` | unset | the server (`.bvh`) and the ray caster (`.rtree`) | Folder for the two disk caches; unset means no file is read or written whatever the flags say |
| `SWARM_SHARE_MESH` | on | `model.cpp` | `0` gives every render object private copies of its mesh and texture (the behaviour before [#8](https://github.com/swarm-subnet/bullet3-swarmfork/pull/8)); on, identical data is shared with reference counts and copy on write, warehouse RSS 658 MB to 228 MB, images identical |

## 6. Build switches

Read by `setup.py` and the Embree script at build time. `wheel.sh` is the published build.

| Variable | Values | Default | What it does |
|---|---|---|---|
| `SWARM_BULLET3_OPT_LEVEL` | `safe`, `v2`, `v3`, `v4` | auto: the highest tier the build CPU and compiler support | `safe` is `-O2`; the others are `-O3 -march=x86-64-<tier> -mtune=native -flto=auto -ffp-contract=off -fno-math-errno -fno-plt`. The published wheel is `v3`: AVX2, FMA, BMI1/2, F16C, LZCNT, MOVBE, Intel Haswell 2013 or newer and AMD Zen. Same source, identical pixels, 1.3x to 1.9x faster steps than `safe` ([#3](https://github.com/swarm-subnet/bullet3-swarmfork/pull/3)) |
| `SWARM_BULLET3_PGO` | `off`, `generate`, `use` | `use` when `pgo_data/` holds profiles, else `off` | Profile-guided optimisation; `wheel.sh` sets `off` |
| `SWARM_BULLET3_CCACHE` | `on`, `off` | `on` when `ccache` is on the PATH | Routes `CC` and `CXX` through ccache after the Embree step; `wheel.sh` sets `off` ([#21](https://github.com/swarm-subnet/bullet3-swarmfork/pull/21)) |
| `SWARM_BULLET3_OPENMP` | `on`, `off` | `on` | `off` builds without `-fopenmp`; the render thread count is then 1 |
| `SWARM_RAYCAST` | `on`, `off` | `on` | `off` builds without Embree and without the ray-cast backend; the flag constants still exist but the ray-cast branches are compiled out |
| `SWARM_BULLET3_EMBREE_CACHE` | a folder | `$XDG_CACHE_HOME/swarm-bullet3/embree`, `~/.cache` when that is unset | Where the compiled Embree is kept, one entry per key (Embree version and tarball hash, both patches, the build script, the compiler); `prefix/` next to the script carries a `KEY` stamp and the script returns at once when it matches ([#21](https://github.com/swarm-subnet/bullet3-swarmfork/pull/21)) |

Embree is 4.4.1 (Apache 2.0, licence vendored), one AVX2 code path, its trees built on one thread at run time (`rtcNewDevice("threads=1")`, the library itself compiles on every core), patched twice: `exact_division.patch` replaces every hardware reciprocal with IEEE division and the build refuses to continue if one remains; `tree_cache.patch` adds save and load of a committed tree as one image. Measured fresh-clone build times on the 12-core build box: 318 s from scratch, 121 s with Embree cached, 70 s with ccache warm as well; the wheel bytes are identical whichever path produced them.

Version lives in `setup.py` (`version=`), distribution name `swarm-bullet3`. The swarm repository pins it in `requirements.txt`; a family can only ask for a switch once the wheel that carries it is published and pinned.

## 7. Tests

Every switch has a Python test under `examples/pybullet/unittests/`, run with the wheel installed:

```bash
cd examples/pybullet/unittests && SWARM_RENDER_THREADS=2 python -m unittest -v <name>
```

| Test | Covers |
|---|---|
| `doubleSidedTest.py` | `VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY`, front and back, colour, depth and mask |
| `materialGroupsTest.py` | `VISUAL_SHAPE_MATERIALS_FROM_MTL`, `URDF_USE_MATERIALS_FROM_MTL` |
| `bvhCacheTest.py` | `GEOM_CONCAVE_BVH_CACHE`: no file without flag or folder, identical hits from a loaded tree, corrupt file rebuilt |
| `sharedMeshTest.py` | mesh and texture sharing gives the same bytes as private copies |
| `textureFilterTest.py` | `ER_TEXTURE_FILTER` |
| `skyTest.py` | `skyHorizonColor`, `skyZenithColor`, gradient orientation, depth untouched |
| `raycastColourTest.py` | `ER_SWARM_RAYCAST` colour: agreement with TinyRenderer, shadow ray, sky, filter, thread counts |
| `shadowMapTest.py` | `ER_SWARM_SHADOW_MAP` against the shadow ray, recasts, thread counts |
| `moverShadowTest.py` | `ER_SWARM_MOVER_SHADOW` |
| `edgeAntialiasTest.py` | `ER_EDGE_ANTIALIAS`, depth and mask unchanged |
| `alphaCutoutTest.py` | `ER_ALPHA_CUTOUT` in colour, depth, shadow ray and shadow map |
| `glintTest.py` | `ER_SPECULAR_GLINT`, `specularColor` reaching the renderer |
| `renderTreeCacheTest.py` | `VISUAL_SHAPE_RENDER_TREE_CACHE` |
| `skySunTest.py` | `ER_SWARM_SKY_SUN`, `skyCloudSeed`, both paths paint the same sky |
| `shadowLightCoeffTest.py` | `shadowLightCoeff` |
| `linearLightTest.py` | `ER_SWARM_LINEAR_LIGHT` |
| `daylightTest.py` | `ER_SWARM_DAYLIGHT`: flag off unchanged, sun to sky ratio, sky light by direction, exposure, photo sky and yaw, haze, soft shadow, a shadow edge with no staircase, leaf light, thread counts |
| `visualMeshTest.py` | `resetMeshData` on a visual mesh: the picture and depth follow the new positions on both colour paths, a wrong vertex count or a body without a mesh raises, two bodies from one mesh move apart, texture and uvs survive, normals are rebuilt, a mover keeps up, the same upload twice is the same bytes, thread counts |
| `glassTest.py` | `VISUAL_SHAPE_GLASS`: the wall behind shows through, no change without daylight, depth and mask keep the pane, tint, grazing reflection, sky through an empty pane, two panes in a row, the bit on `changeVisualShape`, a moved pane, thread counts |

The cross-repository proof that existing families are untouched is the swarm repository's `validator/scripts/verify_render_identity.py`, run on the wheel before and after a change, and its `validator/tests/test_render_backend.py` and `validator/tests/test_sky_sun.py`, which pin ray-cast and sun-sky frames to committed hashes.

## 8. Upstream calls a map builder touches

The rest of PyBullet is upstream and documented in the PyBullet Quickstart Guide, kept in this folder as `pybullet_quickstartguide.pdf`. The calls a Swarm map or family goes through, so a reader knows where the switches above attach:

| Call | Used for | Fork switches it takes |
|---|---|---|
| `createVisualShape`, `createCollisionShape`, `createMultiBody` | every map piece built from an OBJ | visual shape flags, `GEOM_CONCAVE_BVH_CACHE`, `specularColor` |
| `loadURDF` | drones, articulated actors, whole URDF maps | `URDF_USE_MATERIALS_FROM_MTL` |
| `changeVisualShape` | recolouring, hiding (alpha 0), retexturing, the double-sided bit after creation | `VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY`, `specularColor` |
| `getCameraImage`, `getDepthImagesBatch` | every sensor frame | the render flags and the camera arguments above |
| `computeViewMatrix`, `computeProjectionMatrixFOV` | the camera pose and lens | none |
| `setPhysicsEngineParameter` | solver iterations and island size, set by the swarm side at world build | none |
| `setCollisionFilterGroupMask` | static map bodies in an isolated group so terrain never collides with terrain | none |
| `rayTest`, `rayTestBatch` | altitude ray, clearance checks, spawn validation | none |
| `applyExternalForce`, `resetBasePositionAndOrientation` | rotor thrust, wind, moving actors | none |

## 9. Adding a switch

1. Take the next free bit of the enum it belongs to (render flags: 16384) and export the constant in `pybullet.c`.
2. Off by default, and the flag-off path shares no new arithmetic with the flag-on path; run the identity script on the wheel before and after.
3. A test under `examples/pybullet/unittests/` that proves the effect, that depth and mask are untouched where they should be, and that 1, 2 and 4 threads give the same bytes.
4. A row in this file, with the value from the source and the cost you measured; the pull request template asks for it.
