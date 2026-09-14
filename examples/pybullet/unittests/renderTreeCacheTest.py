"""Ray-cast trees of flagged static bodies cached on disk behind VISUAL_SHAPE_RENDER_TREE_CACHE and SWARM_BVH_CACHE_DIR."""
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pybullet as p

NEEDS_BACKEND = unittest.skipUnless(hasattr(p, "ER_SWARM_RAYCAST") and hasattr(p, "VISUAL_SHAPE_RENDER_TREE_CACHE"),
                                    "wheel built without the ray-cast backend or the tree cache")
SIZE = 96
# A bumpy 24 x 24 grid with normals, so the tree has many nodes and the file carries every block.
N = 24
VERTICES = [[x * 0.5, y * 0.5, 0.2 * ((x * 7 + y * 3) % 5)] for y in range(N) for x in range(N)]
NORMALS = [[0.0, 0.0, 1.0]] * len(VERTICES)
INDICES = []
for y in range(N - 1):
  for x in range(N - 1):
    a = y * N + x
    INDICES += [a, a + 1, a + N, a + 1, a + N + 1, a + N]
FLAGS = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | getattr(p, "ER_SWARM_RAYCAST", 0)
CACHE = getattr(p, "VISUAL_SHAPE_RENDER_TREE_CACHE", 0)
LIGHT = [-0.3, 0.6, 0.74]


def build_world(cli, flags, position=(0, 0, 0)):
  """The grid at the given position, flagged or not, plus a red box that always joins the static tree; returns the grid body."""
  grid = p.createVisualShape(p.GEOM_MESH, vertices=VERTICES, indices=INDICES, normals=NORMALS,
                             rgbaColor=[0.8, 0.6, 0.4, 1], flags=flags, physicsClientId=cli)
  body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=grid, basePosition=list(position), physicsClientId=cli)
  box = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1], physicsClientId=cli)
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[6, 6, 1.0], physicsClientId=cli)
  return body


def frame(cli):
  """Colour, depth and segmentation bytes of one shadowed ray-cast frame over the grid."""
  view = p.computeViewMatrix([14, 2, 7], [6, 6, 0.5], [0, 0, 1], physicsClientId=cli)
  proj = p.computeProjectionMatrixFOV(70, 1.0, 0.1, 40.0, physicsClientId=cli)
  _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, proj, shadow=1, lightDirection=LIGHT,
                                           renderer=p.ER_TINY_RENDERER, flags=FLAGS, physicsClientId=cli)
  return (np.asarray(rgb, dtype=np.uint8).tobytes(), np.asarray(depth, dtype=np.float32).tobytes(),
          np.asarray(seg, dtype=np.int32).tobytes())


def render(flags, position=(0, 0, 0)):
  """One frame from a fresh world holding the grid with these visual flags."""
  cli = p.connect(p.DIRECT)
  build_world(cli, flags, position)
  result = frame(cli)
  p.disconnect(physicsClientId=cli)
  return result


def render_after_move(flags):
  """The frame after the grid, already drawn once, is moved one metre along x in a fresh world."""
  cli = p.connect(p.DIRECT)
  body = build_world(cli, flags)
  frame(cli)
  p.resetBasePositionAndOrientation(body, [1.0, 0, 0], [0, 0, 0, 1], physicsClientId=cli)
  result = frame(cli)
  p.disconnect(physicsClientId=cli)
  return result


@NEEDS_BACKEND
class TestRenderTreeCache(unittest.TestCase):
  """Builds the same flagged grid in fresh worlds and checks what the cache folder does to it."""

  def setUp(self):
    """Hand the renderer an empty cache folder."""
    self.cache_dir = tempfile.mkdtemp(prefix="rtreecache")
    os.environ["SWARM_BVH_CACHE_DIR"] = self.cache_dir

  def tearDown(self):
    """Drop the folder and the variable."""
    os.environ.pop("SWARM_BVH_CACHE_DIR", None)
    shutil.rmtree(self.cache_dir, ignore_errors=True)

  def cache_files(self):
    """Every file the renderer left in the cache folder."""
    return sorted(glob.glob(os.path.join(self.cache_dir, "*")))

  def test_no_file_without_flag(self):
    """A grid created without the flag writes nothing."""
    render(0)
    self.assertEqual(self.cache_files(), [])

  def test_no_file_without_cache_dir(self):
    """The flag alone, with no folder set, writes nothing."""
    del os.environ["SWARM_BVH_CACHE_DIR"]
    render(CACHE)
    self.assertEqual(self.cache_files(), [])

  def test_built_then_loaded_are_identical(self):
    """The first world writes one tree file, the second loads it untouched, and both give the same bytes."""
    built = render(CACHE)
    files = self.cache_files()
    self.assertEqual(len(files), 1)
    self.assertTrue(files[0].endswith(".rtree"))
    stamp = (os.path.getsize(files[0]), os.path.getmtime(files[0]))
    loaded = render(CACHE)
    self.assertEqual(built, loaded)
    self.assertEqual((os.path.getsize(files[0]), os.path.getmtime(files[0])), stamp)
    seg = np.frombuffer(built[2], dtype=np.int32)
    self.assertGreater(int((seg == 0).sum()), SIZE)
    self.assertGreater(int((seg == 1).sum()), 0)

  def test_world_tree_draws_the_same_pixels_as_the_static_tree(self):
    """A flagged grid in its own world tree gives the same colour, depth and segmentation bytes as the grid in the static tree."""
    self.assertEqual(render(CACHE), render(0))

  def test_two_builds_write_the_same_bytes(self):
    """Two fresh processes building the same grid write byte-identical files: the build is deterministic."""
    digests = set()
    for _ in range(2):
      folder = tempfile.mkdtemp(prefix="rtreedet")
      env = dict(os.environ, SWARM_BVH_CACHE_DIR=folder)
      digests.add(subprocess.check_output([sys.executable, __file__, "--build"], env=env, text=True).strip())
      shutil.rmtree(folder, ignore_errors=True)
    self.assertEqual(len(digests), 1)

  def test_pose_gets_its_own_file(self):
    """The same mesh at another position is another world tree, so it gets a second file."""
    render(CACHE)
    render(CACHE, position=(1.0, 0, 0))
    self.assertEqual(len(self.cache_files()), 2)

  def test_corrupt_file_is_rebuilt(self):
    """Garbage in the cache file is ignored, rebuilt over, and never changes the bytes."""
    built = render(CACHE)
    path = self.cache_files()[0]
    with open(path, "wb") as f:
      f.write(b"not a tree")
    self.assertEqual(render(CACHE), built)
    self.assertGreater(os.path.getsize(path), 64)
    self.assertEqual(render(CACHE), built)

  def test_moved_body_leaves_its_world_tree(self):
    """A flagged grid that moves after its first frame is drawn like any mover from then on, and writes no second file."""
    self.assertEqual(render_after_move(CACHE), render_after_move(0))
    self.assertNotEqual(render_after_move(CACHE), render(CACHE))
    self.assertEqual(len(self.cache_files()), 1)


def build_hash():
  """Builds the grid once and prints the sha256 of the file it wrote."""
  render(CACHE)
  path = glob.glob(os.path.join(os.environ["SWARM_BVH_CACHE_DIR"], "*.rtree"))[0]
  with open(path, "rb") as f:
    print(hashlib.sha256(f.read()).hexdigest())


if __name__ == '__main__':
  if "--build" in sys.argv:
    build_hash()
  else:
    unittest.main()
