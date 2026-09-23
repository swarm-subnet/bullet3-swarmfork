"""Static instances share canonical mesh trees while preserving surfaces, shadows and thread identity."""
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pybullet as p

from daylightTest import write_obj, write_png
from materialGroupsTest import MTL, OBJ, write_tga

INSTANCED = getattr(p, "VISUAL_SHAPE_RENDER_INSTANCED", 0)
RAY = getattr(p, "ER_SWARM_RAYCAST", 0)
MAP = getattr(p, "ER_SWARM_SHADOW_MAP", 0)
DAY = getattr(p, "ER_SWARM_DAYLIGHT", 0)
LEAF = getattr(p, "ER_SWARM_LEAF_NO_SHADOW", 0)
CUTOUT = getattr(p, "ER_ALPHA_CUTOUT", 0)


def frame(flags=RAY | MAP, shadow=1, eye=(0, 0, 8), target=(0, 0, 0), core=0, size=96):
  """Return all three camera buffers with a fixed light and no mover shadow rays."""
  view = p.computeViewMatrix(eye, target, [0, 1, 0])
  proj = p.computeProjectionMatrixFOV(60, 1, 0.1, 100)
  result = p.getCameraImage(size, size, view, proj, renderer=p.ER_TINY_RENDERER, flags=flags,
                            shadow=shadow, lightDirection=[1, 0, 1], lightAmbientCoeff=0.3,
                            lightDiffuseCoeff=0.7, lightSpecularCoeff=0, shadowLightCoeff=0, shadowCoreRadius=core)
  return tuple(np.asarray(a) for a in result[2:])


def scene(folder, instanced, leaf=False, glass=False, scale=(1, 1, 1)):
  """Place a horizontal mesh above a receiver, optionally textured as a leaf or marked glass."""
  p.resetSimulation()
  ground = p.createVisualShape(p.GEOM_BOX, halfExtents=[4, 4, 0.05], rgbaColor=[1, 1, 1, 1])
  p.createMultiBody(0, -1, ground, basePosition=[0, 0, -0.05])
  path = os.path.join(folder, "card.obj")
  write_obj(path, 1)
  flags = INSTANCED if instanced else 0
  if leaf:
    flags |= p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY
  if glass:
    flags |= p.VISUAL_SHAPE_GLASS
  shape = p.createVisualShape(p.GEOM_MESH, fileName=path, flags=flags, meshScale=scale,
                              rgbaColor=[0.5, 0.8, 0.4, 1], specularColor=[0, 0, 0])
  body = p.createMultiBody(0, -1, shape, basePosition=[0, 0, 1.5])
  if leaf:
    photo = np.full((16, 16, 4), 255, dtype=np.uint8)
    photo[4:12, 4:12, 3] = 0
    texture = os.path.join(folder, "leaf.png")
    write_png(texture, photo)
    p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(texture))
  return body


def grid(path, triangles):
  """Write exactly the requested even triangle count as a tessellated square with shared corners."""
  cells = triangles // 2
  width = 100 if cells >= 100 else cells
  height = cells // width
  with open(path, "w") as out:
    for y in range(height + 1):
      for x in range(width + 1):
        out.write("v %.8f %.8f 0\n" % (x / width, y / height))
    out.write("vn 0 0 1\n")
    for y in range(height):
      for x in range(width):
        a = y * (width + 1) + x + 1
        b, c, d = a + 1, a + width + 2, a + width + 1
        out.write("f %d//1 %d//1 %d//1\nf %d//1 %d//1 %d//1\n" % (a, b, c, a, c, d))


def memory_case(instanced, copies, triangles, scales=1):
  """Measure peak RSS after one render in a fresh process with repeated placements and optional scales."""
  p.connect(p.DIRECT)
  with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "mesh.obj")
    grid(path, triangles)
    shapes = [p.createVisualShape(p.GEOM_MESH, fileName=path, flags=INSTANCED if instanced else 0,
                                  meshScale=[1 + i * 0.125, 1, 1]) for i in range(scales)]
    for i in range(copies):
      p.createMultiBody(0, -1, shapes[i % scales], basePosition=[(i % 20) * 2, (i // 20) * 2, 0])
    frame(flags=RAY, shadow=0, eye=(20, 10, 50), target=(20, 10, 0), size=256)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
  p.disconnect()
  return peak


def child(mode, *args, threads="4", share="1"):
  """Run an isolated measurement so peak RSS and the render thread setting start fresh."""
  env = dict(os.environ, SWARM_RENDER_THREADS=threads, SWARM_SHARE_MESH=share)
  output = subprocess.check_output([sys.executable, __file__, mode] + list(map(str, args)), env=env, text=True)
  return json.loads(output.strip().splitlines()[-1])


@unittest.skipUnless(INSTANCED and RAY, "wheel without static instancing")
class TestInstancedStatic(unittest.TestCase):
  """Check opt-in sharing against the static path and exercise placement lifecycle changes."""

  def setUp(self):
    """Create an isolated client and temporary assets."""
    p.connect(p.DIRECT)
    self.temp = tempfile.TemporaryDirectory()
    self.folder = self.temp.name

  def tearDown(self):
    """Release the client and its assets."""
    p.disconnect()
    self.temp.cleanup()

  def assertFramesEqual(self, left, right):
    """Compare colour, float depth and segmentation as bytes."""
    for name, a, b in zip(("colour", "depth", "mask"), left, right):
      self.assertTrue(a.tobytes() == b.tobytes(), "%s: %d differing values" % (name, np.count_nonzero(a != b)))

  def test_one_body_has_identical_bytes(self):
    """A mesh at the origin has identical colour, depth and mask with the flag off and on."""
    frames = []
    for flag in (0, INSTANCED):
      p.resetSimulation()
      path = os.path.join(self.folder, "one.obj")
      write_obj(path, 1)
      shape = p.createVisualShape(p.GEOM_MESH, fileName=path, flags=flag)
      p.createMultiBody(0, -1, shape)
      frames.append(frame(shadow=0))
    self.assertFramesEqual(*frames)

  def test_shadow_map_and_shadow_ray_match_static(self):
    """A flagged occluder casts the same shadow without enabling mover shadow rays."""
    for flags in (RAY, RAY | MAP, RAY | MAP | DAY):
      scene(self.folder, False)
      reference = frame(flags)
      scene(self.folder, True)
      self.assertFramesEqual(reference, frame(flags))
      self.assertFalse(np.array_equal(reference[0], frame(flags, shadow=0)[0]))

  def test_leaf_cutout_and_no_shadow_match_static(self):
    """Cut-outs, back faces and leaf shadow suppression keep the static surface's bytes."""
    for flags in (RAY | CUTOUT, RAY | MAP | CUTOUT | DAY, RAY | MAP | CUTOUT | DAY | LEAF):
      for eye in ((0, 0, 8), (0, 0, 1)):
        scene(self.folder, False, leaf=True)
        reference = frame(flags, eye=eye, target=(0, 0, 1.5))
        scene(self.folder, True, leaf=True)
        self.assertFramesEqual(reference, frame(flags, eye=eye, target=(0, 0, 1.5)))

  def test_shadow_map_sees_single_sided_back_faces(self):
    """A roof facing away from the light still casts into the map like the static roof."""
    frames = []
    for instanced in (False, True):
      body = scene(self.folder, instanced)
      p.resetBasePositionAndOrientation(body, [0, 0, 1.5], [1, 0, 0, 0])
      frames.append(frame())
    self.assertFramesEqual(*frames)
    self.assertFalse(np.array_equal(frames[1][0], frame(shadow=0)[0]))

  def test_core_map_and_depth_only(self):
    """The fine daylight shadow grid and depth-only rendering agree with static geometry."""
    scene(self.folder, False, leaf=True)
    reference = frame(RAY | MAP | DAY | CUTOUT, core=3)
    scene(self.folder, True, leaf=True)
    self.assertFramesEqual(reference, frame(RAY | MAP | DAY | CUTOUT, core=3))
    depth_only = frame(RAY | CUTOUT | p.ER_DEPTH_ONLY)
    self.assertEqual(reference[1].tobytes(), depth_only[1].tobytes())

  def test_retexturing_recasts_the_map(self):
    """Changing an alpha texture refreshes the shadow cells even when both textures have holes."""
    body = scene(self.folder, True, leaf=True)
    before = frame(RAY | MAP | CUTOUT)
    photo = np.full((16, 16, 4), 255, dtype=np.uint8)
    photo[:, :12, 3] = 0
    texture = os.path.join(self.folder, "new_leaf.png")
    write_png(texture, photo)
    p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(texture))
    after = frame(RAY | MAP | CUTOUT)
    receiver = (before[2] == 0) & (after[2] == 0)
    self.assertFalse(np.array_equal(before[0][receiver], after[0][receiver]))
    self.assertFramesEqual(after, frame(RAY | MAP | CUTOUT))

  def test_glass_matches_static(self):
    """Daylight glass transmits the receiver and retains depth and mask."""
    scene(self.folder, False, glass=True)
    reference = frame(RAY | MAP | DAY)
    scene(self.folder, True, glass=True)
    self.assertFramesEqual(reference, frame(RAY | MAP | DAY))

  def test_move_hide_remove_and_reset(self):
    """Placement updates invalidate the map, and removal or reset leaves no ghost geometry."""
    body = scene(self.folder, True)
    original = frame()
    p.resetBasePositionAndOrientation(body, [2, 0, 1.5], [0, 0, 0, 1])
    moved = frame()
    self.assertFalse(np.array_equal(original[2], moved[2]))
    p.resetBasePositionAndOrientation(body, [0, 0, 1.5], [0, 0, 0, 1])
    self.assertFramesEqual(original, frame())
    p.changeVisualShape(body, -1, rgbaColor=[0.5, 0.8, 0.4, 0])
    hidden = frame()
    self.assertFalse(np.array_equal(original[0], hidden[0]))
    p.changeVisualShape(body, -1, rgbaColor=[0.5, 0.8, 0.4, 1])
    self.assertFramesEqual(original, frame())
    p.removeBody(body)
    removed = frame()
    self.assertFalse(np.any(removed[2] == body))
    scene(self.folder, True)
    self.assertFramesEqual(original, frame())

  def test_nonuniform_and_mirrored_scale(self):
    """Axis-aligned scaled and mirrored cards retain the static coverage, depth and lighting."""
    for scale in ((2, 0.5, 1), (-2, 0.5, 1)):
      scene(self.folder, False, leaf=True, scale=scale)
      reference = frame(RAY | MAP | DAY | CUTOUT)
      scene(self.folder, True, leaf=True, scale=scale)
      self.assertFramesEqual(reference, frame(RAY | MAP | DAY | CUTOUT))

  def test_inverse_transpose_normal(self):
    """A sloped normal under non-uniform scale lights like an explicitly corrected world mesh."""
    vertices = [[-1, -1, -1], [1, -1, 1], [1, 1, 1], [-1, 1, -1]]
    results = []
    for canonical in (False, True):
      p.resetSimulation()
      positions = vertices if canonical else [[v[0] * 2, v[1], v[2] * 0.5] for v in vertices]
      normal = [-1, 0, 1] if canonical else [-0.5, 0, 2]
      shape = p.createVisualShape(p.GEOM_MESH, vertices=positions, indices=[0, 1, 2, 0, 2, 3],
                                  normals=[normal] * 4, meshScale=[2, 1, 0.5] if canonical else [1, 1, 1],
                                  flags=INSTANCED if canonical else 0, rgbaColor=[1, 1, 1, 1])
      p.createMultiBody(0, -1, shape)
      results.append(frame(RAY | DAY, shadow=0))
    self.assertGreater(np.count_nonzero(results[0][2] >= 0), 100)
    self.assertTrue(np.array_equal(results[0][0], results[1][0]))
    self.assertTrue(np.array_equal(results[0][2], results[1][2]))
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=0, atol=2e-7)

  def test_material_groups_keep_their_surfaces(self):
    """Each existing MTL group becomes an instance with its own material and the same body mask."""
    path = Path(self.folder) / "pair.obj"
    path.write_text(OBJ)
    (Path(self.folder) / "pair.mtl").write_text(MTL)
    write_tga(os.path.join(self.folder, "green.tga"), (0, 255, 0))
    frames = []
    for flag in (0, INSTANCED):
      p.resetSimulation()
      shape = p.createVisualShape(p.GEOM_MESH, fileName=str(path), flags=flag | p.VISUAL_SHAPE_MATERIALS_FROM_MTL)
      p.createMultiBody(0, -1, shape)
      frames.append(frame(RAY, shadow=0, eye=(0, -4, 2)))
    self.assertFramesEqual(*frames)
    rgb = frames[1][0][:, :, :3]
    self.assertGreater(np.count_nonzero(rgb[:, :, 0] > rgb[:, :, 1]), 50)
    self.assertGreater(np.count_nonzero(rgb[:, :, 1] > rgb[:, :, 0]), 50)

  def test_deformation_does_not_rewrite_other_instances(self):
    """Rewriting one shared mesh leaves the other placement and its tree untouched."""
    vertices = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]]
    shape = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=[0, 1, 2, 0, 2, 3],
                                normals=[[0, 0, 1]] * 4, flags=INSTANCED)
    first = p.createMultiBody(0, -1, shape, basePosition=[-1, 0, 0])
    p.createMultiBody(0, -1, shape, basePosition=[1, 0, 0])
    before = frame(shadow=0)
    p.resetMeshData(first, [[v[0], v[1], v[2] + 1] for v in vertices])
    after = frame(shadow=0)
    self.assertFalse(np.array_equal(before[1], after[1]))
    for old, new in zip(before, after):
      self.assertEqual(old[:, 48:].tobytes(), new[:, 48:].tobytes())

  def test_visual_frame_and_raster_placement(self):
    """Deferred mesh scale and visual translation also place the mesh on the raster path."""
    path = os.path.join(self.folder, "offset.obj")
    write_obj(path, 0.5)
    for flags in (0, RAY):
      frames = []
      for flag in (0, INSTANCED):
        p.resetSimulation()
        shape = p.createVisualShape(p.GEOM_MESH, fileName=path, flags=flag, meshScale=[2, 1, 1],
                                    visualFramePosition=[1, 0, 0.5])
        p.createMultiBody(0, -1, shape)
        frames.append(frame(flags, shadow=0))
      self.assertFramesEqual(*frames)

  def test_many_copies_use_less_memory(self):
    """Repeated geometry uses much less peak memory when the local tree is shared."""
    plain = child("memory", 0, 50, 6000)
    shared = child("memory", 1, 50, 6000)
    self.assertGreater(plain - shared, 30 * 1024 * 1024, (plain, shared))
    self.assertLess(shared, plain * 0.85, (plain, shared))

  def test_five_scales_share_the_tree(self):
    """Five distinct scale placements add no triangle-sized allocation compared with one scale."""
    one = child("memory", 1, 50, 6000, 1)
    five = child("memory", 1, 50, 6000, 5)
    self.assertLess(five - one, 12 * 1024 * 1024, (one, five))

  def test_threads_give_the_same_bytes(self):
    """Fresh processes render identical colour, depth and mask at one, two and four threads."""
    hashes = [child("identity", threads=t) for t in ("1", "2", "4")]
    self.assertEqual(hashes[0], hashes[1])
    self.assertEqual(hashes[0], hashes[2])


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "memory":
    print(json.dumps(memory_case(*map(int, sys.argv[2:]))))
  elif len(sys.argv) > 1 and sys.argv[1] == "identity":
    p.connect(p.DIRECT)
    with tempfile.TemporaryDirectory() as folder:
      scene(folder, True, leaf=True)
      buffers = frame(RAY | MAP | DAY | CUTOUT | LEAF | p.ER_EDGE_ANTIALIAS | p.ER_TEXTURE_FILTER)
      print(json.dumps(hashlib.sha256(b"".join(a.tobytes() for a in buffers)).hexdigest()))
    p.disconnect()
  else:
    unittest.main()
